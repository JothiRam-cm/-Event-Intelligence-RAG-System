import pandas as pd
from langchain_core.documents import Document


def safe_str(value):
    """Safely convert values to string"""
    if pd.isna(value):
        return "not available"
    return str(value)


def build_event_text(row: pd.Series) -> str:
    return f"""
Event ID {safe_str(row.get('EVENT_ID'))} corresponds to alarm {safe_str(row.get('ALARM_ID'))}
named "{safe_str(row.get('ALARM_NAME'))}".

The event belongs to the category "{safe_str(row.get('CATEGORY_NAME'))}".

The alarm was generated at {safe_str(row.get('ALARM_GENERATED_TIME'))}
and the event occurred at {safe_str(row.get('EVENT_OCCURRENCE_TIME'))}.

The alarm was triggered by component ID {safe_str(row.get('COMPONENT_ID'))}.

Location details include site "{safe_str(row.get('SITE_NAME'))}",
jurisdiction "{safe_str(row.get('JURISDICTION_NAME'))}",
and station "{safe_str(row.get('STATION_NAME'))}".

The current event status is "{safe_str(row.get('EVENT_STATUS'))}"
and the alarm status is "{safe_str(row.get('ALARM_STATUS'))}".

Severity level is "{safe_str(row.get('SEVERITY'))}" with urgency "{safe_str(row.get('URGENCY'))}"
and priority "{safe_str(row.get('PRIORITY'))}".

The escalation count recorded is {safe_str(row.get('BPM_ESCULATION_COUNT'))}.

Primary agency involved is "{safe_str(row.get('PRIMARY_AGENCY'))}"
and secondary agency involved is "{safe_str(row.get('SECONDARY_AGENCY'))}".

The standard operating procedure followed is "{safe_str(row.get('SOP_NAME'))}".
SOP description: {safe_str(row.get('SOP_DESCRIPTION'))}.

Total SOP activities count is {safe_str(row.get('SOP_TOTAL_ACTIVITIES_COUNT'))}
and completed SOP activities count is {safe_str(row.get('SOP_COMPLETED_ACTIVITIES_COUNT'))}.

The SOP document reference is {safe_str(row.get('SOP_DOCUMENT_URL'))}.
""".strip()

def normalize_component_id(value):
    if pd.isna(value):
        return "not available"
    try:
        return str(int(float(value)))
    except:
        return "not available"

def build_metadata(row: pd.Series) -> dict:
    return {
        "event_id": safe_str(row.get("EVENT_ID")),
        "alarm_id": safe_str(row.get("ALARM_ID")),
        "category": safe_str(row.get("CATEGORY_NAME")),
        "component_id": normalize_component_id(row.get("COMPONENT_ID")),
        "severity": safe_str(row.get("SEVERITY")),
        "urgency": safe_str(row.get("URGENCY")),
        "priority": safe_str(row.get("PRIORITY")),
        "site_name": safe_str(row.get("SITE_NAME")),
        "jurisdiction": safe_str(row.get("JURISDICTION_NAME")),
        "station": safe_str(row.get("STATION_NAME")),
        "event_status": safe_str(row.get("EVENT_STATUS")),
        "event_time": safe_str(row.get("EVENT_CREATED_TIME")),
    }

def create_event_documents(df: pd.DataFrame):
    """
    Convert dataframe rows into LangChain Documents.
    """
    documents = []

    for _, row in df.iterrows():
        event_text = build_event_text(row)
        metadata = build_metadata(row)

        documents.append(
            Document(
                page_content=event_text,
                metadata=metadata
            )
        )

    print(f"✅ Created {len(documents)} event documents")
    return documents

# ## test

# if __name__ == "__main__":
#     from load_data import load_event_data

#     df = load_event_data()
#     docs = create_event_documents(df)

#     print("\n--- SAMPLE EVENT TEXT ---\n")
#     print(docs[0].page_content)
#     print("\n--- METADATA ---\n")
#     print(docs[0].metadata)