import html
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

import feedparser
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "investigative_workflow.db"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "lead_classifier.joblib"

st.set_page_config(
    page_title="Investigative Workflow System",
    page_icon="🕵🏾",
    layout="wide",
)


# -----------------------------
# Database
# -----------------------------


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def initialise_database():
    with get_connection() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT,
                source_type TEXT,
                url TEXT,
                date_observed TEXT,
                location TEXT,
                institution TEXT,
                topic TEXT,
                affected_group TEXT,
                public_interest INTEGER DEFAULT 1,
                accountability INTEGER DEFAULT 1,
                impact INTEGER DEFAULT 1,
                evidence INTEGER DEFAULT 1,
                novelty INTEGER DEFAULT 1,
                feasibility INTEGER DEFAULT 1,
                verification_risk INTEGER DEFAULT 1,
                score INTEGER DEFAULT 0,
                priority_band TEXT,
                verification_status TEXT DEFAULT 'Unverified',
                workflow_status TEXT DEFAULT 'New',
                next_action TEXT,
                evidence_notes TEXT,
                right_of_reply TEXT,
                assigned_user TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                description TEXT NOT NULL,
                reference TEXT,
                status TEXT DEFAULT 'Needed',
                notes TEXT,
                created_at TEXT,
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS reply_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER NOT NULL,
                contact_name TEXT,
                institution TEXT,
                contact_date TEXT,
                method TEXT,
                response_status TEXT DEFAULT 'Not contacted',
                response_notes TEXT,
                created_at TEXT,
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                actor TEXT DEFAULT 'Prototype user',
                created_at TEXT,
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            )
            """
        )
        existing_columns = {
            row[1] for row in con.execute("PRAGMA table_info(leads)").fetchall()
        }
        for column, definition in [
            ("training_label", "TEXT"),
            ("training_note", "TEXT"),
        ]:
            if column not in existing_columns:
                con.execute(f"ALTER TABLE leads ADD COLUMN {column} {definition}")
        con.commit()


def write_audit(con, lead_id, action, details, actor="Prototype user"):
    con.execute(
        """
        INSERT INTO audit_logs (lead_id, action, details, actor, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            lead_id,
            action,
            details,
            actor,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def insert_lead(record):
    with get_connection() as con:
        cursor = con.execute(
            """
            INSERT INTO leads (
                title, description, source, source_type, url, date_observed,
                location, institution, topic, affected_group,
                public_interest, accountability, impact, evidence, novelty,
                feasibility, verification_risk, score, priority_band,
                verification_status, workflow_status, next_action,
                evidence_notes, right_of_reply, assigned_user,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(record.values()),
        )
        lead_id = cursor.lastrowid
        write_audit(con, lead_id, "Lead created", record.get("title", "New lead"))
        con.commit()
        return lead_id


def load_leads():
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM leads ORDER BY updated_at DESC", con
        )


def load_classifier():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def predict_lead(model, title, description):
    if model is None:
        return None, None
    combined_text = f"{title} {description}".strip()
    label = model.predict([combined_text])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([combined_text])[0]
        confidence = float(max(probabilities))
    return label, confidence


def load_evidence_items(lead_id):
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM evidence_items WHERE lead_id = ? ORDER BY created_at DESC",
            con,
            params=(lead_id,),
        )


def insert_evidence_item(lead_id, evidence_type, description, reference, status, notes):
    with get_connection() as con:
        con.execute(
            """
            INSERT INTO evidence_items
            (lead_id, evidence_type, description, reference, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_id,
                evidence_type,
                description,
                reference,
                status,
                notes,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        write_audit(
            con,
            lead_id,
            "Evidence added",
            f"{evidence_type}: {description} ({status})",
        )
        con.commit()


def load_reply_contacts(lead_id):
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM reply_contacts WHERE lead_id = ? ORDER BY created_at DESC",
            con,
            params=(lead_id,),
        )


def load_audit_history(lead_id):
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM audit_logs WHERE lead_id = ? ORDER BY created_at DESC",
            con,
            params=(lead_id,),
        )


def load_all_evidence_items():
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM evidence_items ORDER BY created_at DESC", con
        )


def load_all_reply_contacts():
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM reply_contacts ORDER BY created_at DESC", con
        )


def load_all_audit_logs():
    with get_connection() as con:
        return pd.read_sql_query(
            "SELECT * FROM audit_logs ORDER BY created_at DESC", con
        )


def insert_reply_contact(
    lead_id, contact_name, institution, contact_date, method, response_status, response_notes
):
    with get_connection() as con:
        con.execute(
            """
            INSERT INTO reply_contacts
            (lead_id, contact_name, institution, contact_date, method,
             response_status, response_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_id,
                contact_name,
                institution,
                contact_date,
                method,
                response_status,
                response_notes,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        write_audit(
            con,
            lead_id,
            "Right-of-reply record added",
            f"{institution}: {response_status}",
        )
        con.commit()


def update_lead(lead_id, updates):
    assignments = ", ".join(f"{key} = ?" for key in updates)
    details = "; ".join(f"{key} changed to {value}" for key, value in updates.items())
    with get_connection() as con:
        con.execute(
            f"UPDATE leads SET {assignments}, updated_at = ? WHERE id = ?",
            list(updates.values()) + [datetime.now().isoformat(timespec="seconds"), lead_id],
        )
        write_audit(con, lead_id, "Lead updated", details)
        con.commit()


def clean_feed_text(value):
    value = html.unescape(value or "")
    return re.sub(r"<[^>]+>", "", value).strip()


def feed_entry_date(entry):
    parsed = entry.get("published", entry.get("updated", ""))
    return parsed[:10] if parsed else str(date.today())


def feed_url_exists(url):
    if not url:
        return False
    with get_connection() as con:
        row = con.execute("SELECT 1 FROM leads WHERE url = ? LIMIT 1", (url,)).fetchone()
    return row is not None


def import_feed_entry(entry, feed_name):
    title = clean_feed_text(entry.get("title", "Untitled lead"))
    description = clean_feed_text(
        entry.get("summary", entry.get("description", "No summary available."))
    )
    url = entry.get("link", "")
    now = datetime.now().isoformat(timespec="seconds")

    # Feed imports remain deliberately conservative until a journalist reviews them.
    values = {
        "public_interest": 2,
        "accountability": 2,
        "impact": 2,
        "evidence": 1,
        "novelty": 2,
        "feasibility": 3,
        "verification_risk": 3,
    }
    risk = values.pop("verification_risk")
    score = calculate_score(**values, risk=risk)
    band = priority_band(score)
    record = {
        "title": title,
        "description": description or "No summary available.",
        "source": feed_name,
        "source_type": "News site",
        "url": url,
        "date_observed": feed_entry_date(entry),
        "location": "",
        "institution": "",
        "topic": "Other",
        "affected_group": "",
        **values,
        "verification_risk": risk,
        "score": score,
        "priority_band": band,
        "verification_status": "Unverified",
        "workflow_status": "New",
        "next_action": "Review source, accountability relevance, and evidence before pursuit.",
        "evidence_notes": "Imported from RSS; manual review required.",
        "right_of_reply": "",
        "assigned_user": "",
        "created_at": now,
        "updated_at": now,
    }
    insert_lead(record)


# -----------------------------
# Scoring and prompts
# -----------------------------


def calculate_score(public_interest, accountability, impact, evidence, novelty, feasibility, risk):
    return (
        2 * public_interest
        + 2 * accountability
        + impact
        + evidence
        + novelty
        + feasibility
        - 2 * risk
    )


def priority_band(score):
    if score >= 25:
        return "High"
    if score >= 15:
        return "Medium"
    return "Low"


def recommended_action(band, risk, evidence):
    if risk >= 4:
        return "Verify source, evidence, and right-of-reply requirements before pursuit."
    if band == "High" and evidence >= 3:
        return "Assign preliminary investigation and prepare an evidence plan."
    if band == "High":
        return "Gather primary evidence before assigning a full investigation."
    if band == "Medium":
        return "Conduct initial checks and monitor for supporting evidence."
    return "Monitor, archive, or close unless new evidence increases priority."


def score_explanation(values):
    explanations = []
    if values["public_interest"] >= 4:
        explanations.append("high public-interest signal")
    if values["accountability"] >= 4:
        explanations.append("strong institutional-accountability signal")
    if values["impact"] >= 4:
        explanations.append("potentially serious public impact")
    if values["evidence"] <= 2:
        explanations.append("limited evidence currently available")
    if values["verification_risk"] >= 4:
        explanations.append("high verification risk")
    if values["novelty"] >= 4:
        explanations.append("possible novelty or overlooked angle")
    if values["feasibility"] <= 2:
        explanations.append("limited immediate feasibility")
    return "; ".join(explanations) or "No dominant signal; human review is required."


# -----------------------------
# Interface helpers
# -----------------------------


def rating(label, default=3, help_text=None):
    return st.slider(label, min_value=1, max_value=5, value=default, help=help_text)


def show_lead_table(df):
    if df.empty:
        st.info("No lead records have been created yet.")
        return

    display_columns = [
        "id", "title", "topic", "institution", "priority_band",
        "score", "verification_status", "workflow_status", "updated_at"
    ]
    st.dataframe(
        df[display_columns],
        width="stretch",
        hide_index=True,
    )


# -----------------------------
# Application
# -----------------------------


initialise_database()
leads = load_leads()
classifier = load_classifier()

st.title("AI-Assisted Investigative Workflow System")
st.caption(
    "Prototype for prioritising and organising English-language Ghanaian public-accountability leads"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Add Lead",
        "Import RSS",
        "Review Leads",
        "Lead Detail",
        "Label Dataset",
        "Dashboard",
        "Export Data",
    ],
)

if page == "Overview":
    st.subheader("System purpose")
    st.write(
        "This prototype helps journalists and editors record possible public-accountability "
        "leads, assess them using visible criteria, and plan verification. It does not prove "
        "wrongdoing or replace editorial judgement."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total leads", len(leads))
    col2.metric("High priority", int((leads["priority_band"] == "High").sum()) if not leads.empty else 0)
    col3.metric("Unverified", int((leads["verification_status"] == "Unverified").sum()) if not leads.empty else 0)
    col4.metric("Open leads", int((leads["workflow_status"] != "Closed").sum()) if not leads.empty else 0)
    st.write(
        "**Classifier status:** " + ("Trained model available" if classifier is not None else "Not trained yet")
    )

    st.subheader("Workflow")
    st.markdown(
        "1. Record a lead from a public source, document, tip, or manual observation.\n"
        "2. Rate public interest, accountability, impact, evidence, novelty, feasibility, and risk.\n"
        "3. Review the transparent priority recommendation.\n"
        "4. Choose a verification or investigation action.\n"
        "5. Track evidence, right of reply, assignment, and workflow status."
    )

    st.warning(
        "A priority score is a recommendation for human review. It is not a finding of fact."
    )

elif page == "Add Lead":
    st.subheader("Record a new investigative lead")

    with st.form("lead_form"):
        title = st.text_input("Lead title *")
        description = st.text_area("Initial claim, report, tip, or observation *", height=160)

        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source")
            source_type = st.selectbox(
                "Source type",
                ["News site", "Official source", "Public document", "Tip", "Social post", "Blog", "Other"],
            )
            url = st.text_input("URL or source reference")
            date_observed = st.date_input("Date observed", value=date.today())
            location = st.text_input("Location")
            institution = st.text_input("Institution or organisation")

        with col2:
            topic = st.selectbox(
                "Main topic",
                [
                    "Public finance",
                    "Procurement",
                    "Public service",
                    "Health",
                    "Education",
                    "Infrastructure",
                    "Environment",
                    "Policy implementation",
                    "Institutional conduct",
                    "Other",
                ],
            )
            affected_group = st.text_input("Affected person or group")
            assigned_user = st.text_input("Assigned reporter or editor")
            verification_status = st.selectbox(
                "Initial verification status",
                ["Unverified", "Partly corroborated", "Corroborated", "Closed"],
            )
            workflow_status = st.selectbox(
                "Workflow status",
                ["New", "Verify first", "Assigned", "Monitoring", "Editorial review", "Closed"],
            )

        st.markdown("### Lead assessment")
        st.caption("Use the available information only. Ratings can be revised after new evidence appears.")
        c1, c2, c3 = st.columns(3)
        with c1:
            public_interest = rating("Public interest", 3)
            accountability = rating("Accountability relevance", 3)
            impact = rating("Potential impact", 3)
        with c2:
            evidence = rating("Evidence availability", 2)
            novelty = rating("Novelty", 3)
            feasibility = rating("Feasibility", 3)
        with c3:
            verification_risk = rating("Verification risk", 3)
            st.info("Higher verification risk reduces the priority score.")

        evidence_notes = st.text_area("Known evidence and evidence gaps", height=100)
        right_of_reply = st.text_area("Right-of-reply notes", height=80)
        submitted = st.form_submit_button("Calculate and save lead")

    if submitted:
        if not title.strip() or not description.strip():
            st.error("Lead title and description are required.")
        else:
            score = calculate_score(
                public_interest,
                accountability,
                impact,
                evidence,
                novelty,
                feasibility,
                verification_risk,
            )
            band = priority_band(score)
            values = {
                "public_interest": public_interest,
                "accountability": accountability,
                "impact": impact,
                "evidence": evidence,
                "novelty": novelty,
                "feasibility": feasibility,
                "verification_risk": verification_risk,
            }
            action = recommended_action(band, verification_risk, evidence)
            now = datetime.now().isoformat(timespec="seconds")
            record = {
                "title": title.strip(),
                "description": description.strip(),
                "source": source.strip(),
                "source_type": source_type,
                "url": url.strip(),
                "date_observed": str(date_observed),
                "location": location.strip(),
                "institution": institution.strip(),
                "topic": topic,
                "affected_group": affected_group.strip(),
                "public_interest": public_interest,
                "accountability": accountability,
                "impact": impact,
                "evidence": evidence,
                "novelty": novelty,
                "feasibility": feasibility,
                "verification_risk": verification_risk,
                "score": score,
                "priority_band": band,
                "verification_status": verification_status,
                "workflow_status": workflow_status,
                "next_action": action,
                "evidence_notes": evidence_notes.strip(),
                "right_of_reply": right_of_reply.strip(),
                "assigned_user": assigned_user.strip(),
                "created_at": now,
                "updated_at": now,
            }
            insert_lead(record)
            st.success(f"Lead saved as {band.lower()} priority with score {score}.")
            st.write("**Reasoning:**", score_explanation(values))
            st.write("**Suggested next action:**", action)

elif page == "Import RSS":
    st.subheader("Import public RSS feed items")
    st.caption("Imported items are unverified and must be assessed by a journalist or editor.")

    feed_url = st.text_input(
        "RSS feed URL",
        placeholder="Paste the RSS address of an accessible public news feed",
    )
    feed_name = st.text_input("Feed or source name", value="Public RSS feed")
    load_feed = st.button("Load feed")

    if load_feed:
        if not feed_url.strip():
            st.error("Enter an RSS feed URL.")
        else:
            parsed = feedparser.parse(feed_url.strip())
            if getattr(parsed, "bozo", False) and not parsed.entries:
                st.error("The feed could not be read. Check the address and try again.")
            elif not parsed.entries:
                st.warning("The feed was read but contained no entries.")
            else:
                st.session_state["rss_entries"] = parsed.entries
                st.session_state["rss_feed_name"] = feed_name.strip() or "Public RSS feed"
                st.success(f"Loaded {len(parsed.entries)} feed items.")

    entries = st.session_state.get("rss_entries", [])
    if entries:
        rows = []
        for index, entry in enumerate(entries):
            rows.append(
                {
                    "index": index,
                    "title": clean_feed_text(entry.get("title", "Untitled")),
                    "date": feed_entry_date(entry),
                    "link": entry.get("link", ""),
                    "summary": clean_feed_text(entry.get("summary", entry.get("description", ""))),
                }
            )
        feed_df = pd.DataFrame(rows)
        st.dataframe(feed_df[["index", "title", "date", "link"]], hide_index=True, width="stretch")

        choices = {
            f"{row['index']}: {row['title'][:100]}": row["index"]
            for _, row in feed_df.iterrows()
            if not feed_url_exists(row["link"])
        }
        selected_labels = st.multiselect(
            "Select new items to import",
            list(choices.keys()),
        )
        if st.button("Import selected items"):
            imported = 0
            for label in selected_labels:
                entry = entries[choices[label]]
                if not feed_url_exists(entry.get("link", "")):
                    import_feed_entry(entry, st.session_state.get("rss_feed_name", "Public RSS feed"))
                    imported += 1
            st.success(f"Imported {imported} new lead(s).")

elif page == "Label Dataset":
    st.subheader("Label leads for the baseline classifier")
    st.caption(
        "Use one label per lead. Labels support model training and do not replace the investigative priority score."
    )

    if leads.empty:
        st.info("Add or import leads before labelling the dataset.")
    else:
        label_counts = leads["training_label"].fillna("Unlabelled").value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        st.dataframe(label_counts, hide_index=True, width="stretch")

        selected_id = st.selectbox("Select lead to label", leads["id"].tolist())
        selected_lead = leads[leads["id"] == selected_id].iloc[0]
        st.markdown(f"### {selected_lead['title']}")
        st.write(selected_lead["description"])

        label_options = [
            "Unlabelled",
            "accountability_relevant",
            "routine",
            "verification_needed",
        ]
        current_label = selected_lead["training_label"] or "Unlabelled"
        current_index = label_options.index(current_label) if current_label in label_options else 0

        with st.form("label_lead_form"):
            training_label = st.selectbox(
                "Training label",
                label_options,
                index=current_index,
            )
            training_note = st.text_area(
                "Reason for label",
                value=selected_lead["training_note"] or "",
            )
            save_label = st.form_submit_button("Save label")

        if save_label:
            update_lead(
                int(selected_id),
                {
                    "training_label": None if training_label == "Unlabelled" else training_label,
                    "training_note": training_note.strip(),
                },
            )
            st.success("Training label saved.")
            st.rerun()

elif page == "Review Leads":
    st.subheader("Review and filter leads")

    if leads.empty:
        st.info("Add a lead first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            band_filter = st.multiselect(
                "Priority band",
                sorted(leads["priority_band"].dropna().unique()),
                default=sorted(leads["priority_band"].dropna().unique()),
            )
        with col2:
            status_filter = st.multiselect(
                "Workflow status",
                sorted(leads["workflow_status"].dropna().unique()),
                default=sorted(leads["workflow_status"].dropna().unique()),
            )
        with col3:
            topic_filter = st.multiselect(
                "Topic",
                sorted(leads["topic"].dropna().unique()),
                default=sorted(leads["topic"].dropna().unique()),
            )

        filtered = leads[
            leads["priority_band"].isin(band_filter)
            & leads["workflow_status"].isin(status_filter)
            & leads["topic"].isin(topic_filter)
        ]
        st.write(f"Showing {len(filtered)} of {len(leads)} leads")
        show_lead_table(filtered)

elif page == "Lead Detail":
    st.subheader("Inspect or update a lead")

    if leads.empty:
        st.info("Add a lead first.")
    else:
        selected_id = st.selectbox("Select lead ID", leads["id"].tolist())
        lead = leads[leads["id"] == selected_id].iloc[0]

        st.markdown(f"### {lead['title']}")
        st.write(lead["description"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Priority", lead["priority_band"])
        col2.metric("Score", int(lead["score"]))
        col3.metric("Verification", lead["verification_status"])

        st.write("**Source:**", lead["source"] or "Not recorded")
        st.write("**Institution:**", lead["institution"] or "Not recorded")
        st.write("**Topic:**", lead["topic"] or "Not recorded")
        st.write("**Affected group:**", lead["affected_group"] or "Not recorded")
        st.write("**Current next action:**", lead["next_action"] or "Not recorded")

        if classifier is not None:
            model_label, model_confidence = predict_lead(
                classifier, lead["title"], lead["description"]
            )
            st.info(
                f"Model suggestion: {model_label}"
                + (f" ({model_confidence:.1%} confidence)" if model_confidence is not None else "")
                + ". This is advisory and does not replace human assessment."
            )

        st.markdown("### Assessment criteria")
        assessment_df = pd.DataFrame(
            {
                "Criterion": [
                    "Public interest", "Accountability relevance", "Potential impact",
                    "Evidence availability", "Novelty", "Feasibility", "Verification risk"
                ],
                "Rating": [
                    lead["public_interest"], lead["accountability"], lead["impact"],
                    lead["evidence"], lead["novelty"], lead["feasibility"], lead["verification_risk"]
                ],
            }
        )
        st.dataframe(assessment_df, hide_index=True, width="stretch")

        with st.form("update_lead_form"):
            st.markdown("### Revise assessment")
            st.caption("Update the ratings when new evidence or context becomes available.")
            r1, r2, r3 = st.columns(3)
            with r1:
                new_public_interest = st.slider("Public interest", 1, 5, int(lead["public_interest"]))
                new_accountability = st.slider("Accountability relevance", 1, 5, int(lead["accountability"]))
                new_impact = st.slider("Potential impact", 1, 5, int(lead["impact"]))
            with r2:
                new_evidence = st.slider("Evidence availability", 1, 5, int(lead["evidence"]))
                new_novelty = st.slider("Novelty", 1, 5, int(lead["novelty"]))
                new_feasibility = st.slider("Feasibility", 1, 5, int(lead["feasibility"]))
            with r3:
                new_risk = st.slider("Verification risk", 1, 5, int(lead["verification_risk"]))
                revised_score = calculate_score(
                    new_public_interest,
                    new_accountability,
                    new_impact,
                    new_evidence,
                    new_novelty,
                    new_feasibility,
                    new_risk,
                )
                revised_band = priority_band(revised_score)
                st.metric("Revised recommendation", revised_band)
                st.metric("Revised score", revised_score)

            new_workflow_status = st.selectbox(
                "Update workflow status",
                ["New", "Verify first", "Assigned", "Monitoring", "Editorial review", "Closed"],
                index=["New", "Verify first", "Assigned", "Monitoring", "Editorial review", "Closed"].index(lead["workflow_status"]),
            )
            new_verification_status = st.selectbox(
                "Update verification status",
                ["Unverified", "Partly corroborated", "Corroborated", "Closed"],
                index=["Unverified", "Partly corroborated", "Corroborated", "Closed"].index(lead["verification_status"]),
            )
            new_action = st.text_area("Next action", value=lead["next_action"] or "")
            new_evidence_notes = st.text_area("Evidence notes", value=lead["evidence_notes"] or "")
            new_reply = st.text_area("Right-of-reply notes", value=lead["right_of_reply"] or "")
            new_assignee = st.text_input("Assigned user", value=lead["assigned_user"] or "")
            update_button = st.form_submit_button("Save update")

        if update_button:
            revised_action = new_action.strip() or recommended_action(
                revised_band,
                new_risk,
                new_evidence,
            )
            update_lead(
                int(selected_id),
                {
                    "public_interest": new_public_interest,
                    "accountability": new_accountability,
                    "impact": new_impact,
                    "evidence": new_evidence,
                    "novelty": new_novelty,
                    "feasibility": new_feasibility,
                    "verification_risk": new_risk,
                    "score": revised_score,
                    "priority_band": revised_band,
                    "workflow_status": new_workflow_status,
                    "verification_status": new_verification_status,
                    "next_action": revised_action,
                    "evidence_notes": new_evidence_notes,
                    "right_of_reply": new_reply,
                    "assigned_user": new_assignee,
                },
            )
            st.success("Lead updated.")
            st.rerun()

        st.markdown("### Evidence checklist")
        evidence_items = load_evidence_items(int(selected_id))
        if evidence_items.empty:
            st.info("No structured evidence items have been added yet.")
        else:
            st.dataframe(
                evidence_items[
                    ["evidence_type", "description", "reference", "status", "notes"]
                ],
                hide_index=True,
                width="stretch",
            )

        with st.form("evidence_item_form"):
            evidence_type = st.selectbox(
                "Evidence item type",
                [
                    "Primary document",
                    "Official statement",
                    "Independent source",
                    "Affected person interview",
                    "Expert interview",
                    "Financial or procurement record",
                    "Field observation",
                    "Other",
                ],
            )
            evidence_description = st.text_input("Evidence description")
            evidence_reference = st.text_input("Link, document reference, or contact reference")
            evidence_status = st.selectbox(
                "Evidence status",
                ["Needed", "Located", "Partly checked", "Verified", "Unavailable"],
            )
            evidence_notes = st.text_area("Evidence notes")
            add_evidence = st.form_submit_button("Add evidence item")

        if add_evidence:
            if not evidence_description.strip():
                st.error("Enter an evidence description.")
            else:
                insert_evidence_item(
                    int(selected_id),
                    evidence_type,
                    evidence_description.strip(),
                    evidence_reference.strip(),
                    evidence_status,
                    evidence_notes.strip(),
                )
                st.success("Evidence item added.")
                st.rerun()

        st.markdown("### Right-of-reply tracker")
        reply_contacts = load_reply_contacts(int(selected_id))
        if reply_contacts.empty:
            st.info("No right-of-reply contact has been recorded yet.")
        else:
            st.dataframe(
                reply_contacts[
                    [
                        "contact_name", "institution", "contact_date", "method",
                        "response_status", "response_notes"
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        with st.form("reply_contact_form"):
            contact_name = st.text_input("Contact person or office")
            contact_institution = st.text_input(
                "Institution contacted", value=lead["institution"] or ""
            )
            contact_date = st.date_input("Contact date", value=date.today())
            contact_method = st.selectbox(
                "Contact method",
                ["Email", "Telephone", "In person", "Letter", "Online form", "Other"],
            )
            response_status = st.selectbox(
                "Response status",
                ["Not contacted", "Contacted - awaiting response", "Response received", "No response", "Declined to comment"],
            )
            response_notes = st.text_area("Response or contact notes")
            add_reply = st.form_submit_button("Add right-of-reply record")

        if add_reply:
            if not contact_institution.strip():
                st.error("Enter the institution contacted.")
            else:
                insert_reply_contact(
                    int(selected_id),
                    contact_name.strip(),
                    contact_institution.strip(),
                    str(contact_date),
                    contact_method,
                    response_status,
                    response_notes.strip(),
                )
                st.success("Right-of-reply record added.")
                st.rerun()

        st.markdown("### Audit history")
        audit_history = load_audit_history(int(selected_id))
        if audit_history.empty:
            st.info("No audit history is available for this lead yet.")
        else:
            st.dataframe(
                audit_history[["created_at", "action", "details", "actor"]],
                hide_index=True,
                width="stretch",
            )

elif page == "Export Data":
    st.subheader("Export research and workflow data")
    st.write(
        "Download the records for analysis, backup, thesis appendices, and future Chapter Four reporting."
    )

    export_leads = load_leads()
    export_evidence = load_all_evidence_items()
    export_replies = load_all_reply_contacts()
    export_audit = load_all_audit_logs()

    export_sets = [
        ("Lead records", export_leads, "lead_records.csv"),
        ("Evidence items", export_evidence, "evidence_items.csv"),
        ("Right-of-reply records", export_replies, "right_of_reply_records.csv"),
        ("Audit history", export_audit, "audit_history.csv"),
    ]

    for label, export_df, filename in export_sets:
        st.markdown(f"### {label}")
        st.write(f"{len(export_df)} record(s)")
        st.download_button(
            label=f"Download {filename}",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )

elif page == "Dashboard":
    st.subheader("Investigative lead dashboard")

    if leads.empty:
        st.info("Add leads to populate the dashboard.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            counts = leads["priority_band"].value_counts().reset_index()
            counts.columns = ["priority_band", "count"]
            fig = px.bar(
                counts,
                x="priority_band",
                y="count",
                color="priority_band",
                title="Leads by priority band",
            )
            st.plotly_chart(fig, width="stretch")
        with col2:
            statuses = leads["workflow_status"].value_counts().reset_index()
            statuses.columns = ["workflow_status", "count"]
            fig = px.pie(
                statuses,
                names="workflow_status",
                values="count",
                title="Workflow status distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Highest-priority leads")
        top_leads = leads.sort_values("score", ascending=False).head(10)
        show_lead_table(top_leads)

        st.markdown("### Verification-risk overview")
        risk_df = leads.groupby("verification_status").size().reset_index(name="count")
        st.dataframe(risk_df, hide_index=True, width="stretch")
