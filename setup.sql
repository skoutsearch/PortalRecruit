-- Database & Schema
CREATE DATABASE IF NOT EXISTS PORTALRECRUIT_DB;
CREATE SCHEMA IF NOT EXISTS PORTALRECRUIT_SCHEMA;
USE DATABASE PORTALRECRUIT_DB;
USE SCHEMA PORTALRECRUIT_SCHEMA;

-- Secret (GitHub Token)
CREATE OR REPLACE SECRET git_secret
  TYPE = password
  USERNAME = 'PortalRecruit'
  PASSWORD = '<GITHUB_TOKEN>';

-- API Integration
CREATE OR REPLACE API INTEGRATION git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/PortalRecruit/')
  ALLOWED_AUTHENTICATION_SECRETS = (git_secret)
  ENABLED = TRUE;

-- Git Repository
CREATE OR REPLACE GIT REPOSITORY portal_recruit_repo
  API_INTEGRATION = git_api_integration
  GIT_CREDENTIALS = git_secret
  ORIGIN = 'https://github.com/PortalRecruit/PortalRecruit.git';

-- Fetch latest code
ALTER GIT REPOSITORY portal_recruit_repo FETCH;

-- Create Streamlit App
CREATE OR REPLACE STREAMLIT portal_recruit_app
  ROOT_LOCATION = '@portal_recruit_repo/branches/main'
  MAIN_FILE = '/src/dashboard/Home.py'
  QUERY_WAREHOUSE = 'COMPUTE_WH';
