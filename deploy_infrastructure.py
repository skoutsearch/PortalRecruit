import toml
import subprocess
import os

# --- CONFIGURATION ---
# Path to your secrets file
TOML_PATH = '/media/jch903/fidelio/CLAUDOG/PortalRecruit/src/dashboard/.streamlit/secrets.toml'

# The APIs your app needs to talk to (Add any others here!)
DOMAINS = ["api.openai.com", "api.sportradar.com", "api.sportradar.us", "github.com"]

# Your Snowflake info
DB = "PORTALRECRUIT_DB"
SCHEMA = "PORTALRECRUIT_SCHEMA"
INTEGRATION_NAME = "PORTAL_RECRUIT_INTEGRATION"
NETWORK_RULE_NAME = "PORTAL_RECRUIT_NETWORK_RULE"

SNOW_CLI = "/home/jch903/.venv_310/bin/snow"


def run_sql(query):
    # Uses the 'snow' CLI you already installed
    cmd = [SNOW_CLI, "sql", "-q", query]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
    else:
        print(f"✅ Success")


print("--- 1. Parsing Secrets TOML ---")
try:
    secrets = toml.load(TOML_PATH)
except FileNotFoundError:
    print(f"❌ Could not find file at {TOML_PATH}")
    exit()

secret_names = []

# Flatten secrets (e.g., [openai] api_key -> OPENAI_API_KEY)
for section, content in secrets.items():
    if isinstance(content, dict):
        for key, value in content.items():
            secret_name = f"{section}_{key}".upper()
            print(f"Creating Secret: {secret_name}...")
            # Create the secret object in Snowflake
            sql = f"""
            CREATE OR REPLACE SECRET {DB}.{SCHEMA}.{secret_name}
            TYPE = GENERIC_STRING
            SECRET_STRING = '{value}';
            """
            run_sql(sql)
            secret_names.append(f"{DB}.{SCHEMA}.{secret_name}")
    else:
        # Handle top-level keys
        secret_name = section.upper()
        print(f"Creating Secret: {secret_name}...")
        sql = f"""
        CREATE OR REPLACE SECRET {DB}.{SCHEMA}.{secret_name}
        TYPE = GENERIC_STRING
        SECRET_STRING = '{content}';
        """
        run_sql(sql)
        secret_names.append(f"{DB}.{SCHEMA}.{secret_name}")

print("--- 2. Creating Network Rules (Firewall) ---")
domain_list = ", ".join([f"'{d}'" for d in DOMAINS])
rule_sql = f"""
CREATE OR REPLACE NETWORK RULE {DB}.{SCHEMA}.{NETWORK_RULE_NAME}
MODE = EGRESS
TYPE = HOST_PORT
VALUE_LIST = ({domain_list});
"""
run_sql(rule_sql)

print("--- 3. Creating External Access Integration ---")
# This bundles the secrets and the network rule together
secrets_list = ", ".join(secret_names)
integration_sql = f"""
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {INTEGRATION_NAME}
ALLOWED_NETWORK_RULES = ({DB}.{SCHEMA}.{NETWORK_RULE_NAME})
ALLOWED_AUTHENTICATION_SECRETS = ({secrets_list})
ENABLED = TRUE;
"""
run_sql(integration_sql)

print("--- 4. Updating Streamlit App Permission ---")
# We have to re-create the app or alter it to use the integration
app_sql = f"""
ALTER STREAMLIT {DB}.{SCHEMA}.PORTAL_RECRUIT_APP
SET EXTERNAL_ACCESS_INTEGRATIONS = ({INTEGRATION_NAME});
"""
run_sql(app_sql)

print("DONE! Infrastructure is live.")
