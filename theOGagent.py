import streamlit as st
from dotenv import load_dotenv
import os
import boto3
import pandas as pd
from PIL import Image




from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import Tool, tool
from langgraph.prebuilt import create_react_agent

# ─── Load env & AWS creds ───────────────────────────────────────────────
load_dotenv()
AWS_CONFIG = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "region_name": os.getenv("AWS_REGION", "us-east-1")
}

# ─── Initialize model ──────────────────────────────────────────────────
model = init_chat_model(
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse"
)

# ─── Define tools ──────────────────────────────────────────────────────
# define tool brooo 
@tool("trusted_advisor_finops", return_direct=True)
def fetch_optimized_trusted_advisor_data():
    """
    Fetches detailed AWS Trusted Advisor cost optimization recommendations.
    Dynamically adapts to check types and includes useful metadata.
    Ideal for FinOps teams and automation agents.
    """
    import boto3
    import pandas as pd

    support_client = boto3.client('support', region_name='us-east-1')
    checks_response = support_client.describe_trusted_advisor_checks(language='en')
    cost_checks = [check for check in checks_response['checks'] if 'cost' in check['category'].lower()]
    
    final_data = []

    for check in cost_checks:
        check_id = check['id']
        check_name = check['name']
        metadata_keys = check.get('metadata', [])

        result = support_client.describe_trusted_advisor_check_result(checkId=check_id, language='en')
        flagged = result['result'].get('flaggedResources', [])

        for res in flagged:
            entry = {
                "Check Name": check_name,
                "Resource ID": res.get('resourceId', 'N/A'),
                "Status": res.get('status', 'N/A'),
            }
            metadata = res.get('metadata', [])
            for i, key in enumerate(metadata_keys):
                entry[key] = metadata[i] if i < len(metadata) else 'N/A'
            final_data.append(entry)

    if final_data:
        df = pd.DataFrame(final_data)

        # Remove constant or redundant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                df.drop(columns=[col], inplace=True)

        # Try numeric sort if savings exist
        saving_cols = [c for c in df.columns if 'saving' in c.lower()]
        if saving_cols:
            df[saving_cols[0]] = pd.to_numeric(
                df[saving_cols[0]].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )
            df = df.sort_values(by=saving_cols[0], ascending=False)

        return df.to_dict(orient="records")


# initialize the AWS Cost Optimization Hub tool (this is other tool, simply testing this tool)
@tool("cost_optimization_hub", return_direct=True)
def fetch_cost_optimization_hub_recommendations():
    """
    Fetches detailed cost optimization recommendations from AWS Cost Optimization Hub.

    This function interacts with the AWS Cost Optimization Hub to retrieve a list of recommendations aimed at reducing costs across various AWS resources. The recommendations include details such as estimated monthly savings, implementation effort, and specific actions to optimize resource usage.

    Returns:
        list: A list of dictionaries containing detailed cost optimization recommendations. Each dictionary includes:
            - Recommendation ID: Unique identifier for the recommendation.
            - Resource Type: Type of AWS resource (e.g., EC2, EBS, RDS).
            - Resource ID: Identifier for the specific resource.
            - Estimated Monthly Savings ($): Potential monthly savings in USD.
            - Estimated Savings Percentage: Percentage of cost savings.
            - Estimated Monthly Cost ($): Current monthly cost of the resource.
            - Implementation Effort: Effort required to implement the recommendation.
            - Is Resource Restart Needed: Indicates if a resource restart is needed.
            - Is Rollback Possible: Indicates if rollback is possible.
            - Top Recommended Action: Suggested action to optimize the resource.
            - Current Resource Summary: Summary of the current resource configuration.
            - Recommended Resource Summary: Summary of the recommended resource configuration.
            - Currency: Currency code for the cost values.

    If no recommendations are found, the function returns a message indicating that no recommendations were found.
    """
    client = boto3.client('cost-optimization-hub', region_name='us-east-1')
    response = client.list_recommendations()
    recommendations = response.get('items', [])
    
    data = []
    for rec in recommendations:
        data.append({
            "Recommendation ID": rec.get('recommendationId', 'N/A'),
            "Resource Type": rec.get('currentResourceType', 'N/A'),
            "Resource ID": rec.get('resourceId', 'N/A'),
            "Estimated Monthly Savings ($)": rec.get('estimatedMonthlySavings', 0.0),
            "Estimated Savings Percentage": rec.get('estimatedSavingsPercentage', 'N/A'),
            "Estimated Monthly Cost ($)": rec.get('estimatedMonthlyCost', 0.0),
            "Implementation Effort": rec.get('implementationEffort', 'N/A'),
            "Is Resource Restart Needed": rec.get('restartNeeded', 'N/A'),
            "Is Rollback Possible": rec.get('rollbackPossible', 'N/A'),
            "Top Recommended Action": rec.get('actionType', 'N/A'),
            "Current Resource Summary": rec.get('currentResourceSummary', 'N/A'),
            "Recommended Resource Summary": rec.get('recommendedResourceSummary', 'N/A'),
            "Currency": rec.get('currencyCode', 'N/A')
        })
    
    if data:
        df = pd.DataFrame(data)
        return df.to_dict(orient="records")
    return "No recommendations found."


# These are my tools brooo
tools = [
    Tool(
        name="trusted_advisor_finops",
        func=fetch_optimized_trusted_advisor_data,
        description="Check‑based cost alerts from AWS Trusted Advisor."
    ),
    Tool(
        name="cost_optimization_hub",
        func=fetch_cost_optimization_hub_recommendations,
        description="Structured, prioritized cost recommendations from AWS Cost Optimization Hub."
    )
]

agent_executor = create_react_agent(model, tools)

# ─── Streamlit UI ───────────────────────────────────────────────────────
st.title("🕵️‍♂️ FinAI Agent Recommendations")

# Agent avatar
# Load and display your agent avatar
agent_img = Image.open(r"C:\1.THE REAL SRUJAN\agent.webp")
st.image(agent_img, width=100, caption="Your FinOps Agent")

default_prompt = """
You are a FinOps strategist with extensive experience in AWS. Analyze cost optimization opportunities using all tools note dont mention or print any resource ids or confdential information :

1. **AWS Trusted Advisor** — For cost-saving opportunities and underutilization warnings
2. **AWS Cost Optimization Hub** — For structured cost recommendations with savings estimates

Provide a comprehensive analysis including:

### 📊 Cost Analysis Summary
- Total identified savings opportunities :
- Number of recommendations by category :
- Key AWS services and resource types responsible for the highest waste:
- Common inefficiency patterns:

### 🔧 Technical Recommendations
- Resource-specific findings
- Current vs. recommended configurations
- Implementation complexity
- Expected cost impact

### 🚀 Service Modernization Plan
Using AWS Services Knowledge:
- Legacy services identified for upgrade
- Recommended modern alternatives new AWS services and technologies
- Migration benefits and considerations
- Architecture improvement suggestions

### 📋 Action Items
- Prioritized list of actions (High/Medium/Low)
- 30/60/90 day implementation plan
- Required teams and skills
- Risk assessment for each change

Focus on practical, actionable insights that combine cost optimization with service modernization opportunities.
"""

if st.button("🚀 Invoke Agent"):
    st.info("Agent invoked — starting analysis…")
    st.write("**Thinking:**.........")
    final_response = ""
    with st.spinner("Analyzing Your AWS Infrastructure..."):
        # stream but only keep the last message
        for step in agent_executor.stream(
            {"messages":[HumanMessage(content=default_prompt)]},
            stream_mode="values"
        ):
            # always overwrite; at the end this is your final output
            content = step["messages"][-1].content
            # ensure it's a string
            final_response = content if isinstance(content, str) else str(content)

    st.success("✅ Analysis complete!")
    st.subheader("📝 Agent Output")
    st.markdown(final_response)