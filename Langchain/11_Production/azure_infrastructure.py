"""
Azure Infrastructure as Code for LangChain Production Deployment

This module provides comprehensive Infrastructure as Code (IaC) examples using
Terraform, ARM templates, and Azure CLI for deploying production-ready
LangChain applications on Azure.

Key components covered:
1. Azure Resource Manager (ARM) templates
2. Terraform configurations
3. Azure CLI automation scripts
4. Bicep templates for modern IaC
5. Production-ready resource configurations
"""

import os
import json
import yaml
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.keyvault import KeyVaultManagementClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.web import WebSiteManagementClient


# ============================================================================
# 1. AZURE RESOURCE CONFIGURATION MODELS
# ============================================================================

@dataclass
class AzureConfig:
    """Azure configuration for LangChain deployment"""
    subscription_id: str
    resource_group_name: str
    location: str = "East US"
    environment: str = "production"
    application_name: str = "langchain-app"
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {
                "Environment": self.environment,
                "Application": self.application_name,
                "ManagedBy": "Terraform",
                "CostCenter": "AI-ML",
                "Owner": "MLOps-Team"
            }

@dataclass
class OpenAIConfig:
    """Azure OpenAI Service configuration"""
    account_name: str
    sku_name: str = "S0"
    kind: str = "OpenAI"
    deployments: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.deployments is None:
            self.deployments = [
                {
                    "name": "gpt-4",
                    "model": "gpt-4",
                    "version": "0613",
                    "capacity": 10
                },
                {
                    "name": "text-embedding-ada-002",
                    "model": "text-embedding-ada-002", 
                    "version": "2",
                    "capacity": 30
                }
            ]

@dataclass
class CosmosDBConfig:
    """Azure Cosmos DB configuration for vector storage"""
    account_name: str
    database_name: str = "langchain_db"
    container_name: str = "embeddings"
    consistency_level: str = "Session"
    throughput: int = 400
    vector_policy: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.vector_policy is None:
            self.vector_policy = {
                "type": "vector",
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 1536
            }


# ============================================================================
# 2. TERRAFORM CONFIGURATION GENERATOR
# ============================================================================

class TerraformConfigGenerator:
    """Generate Terraform configurations for Azure LangChain deployment"""
    
    def __init__(self, azure_config: AzureConfig):
        self.config = azure_config
        
    def generate_providers(self) -> str:
        """Generate Terraform provider configuration"""
        return f'''
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }}
    azuread = {{
      source  = "hashicorp/azuread"
      version = "~> 2.0"
    }}
  }}
  
  backend "azurerm" {{
    resource_group_name  = "{self.config.resource_group_name}-tfstate"
    storage_account_name = "{self.config.application_name.replace('-', '')}tfstate"
    container_name       = "tfstate"
    key                  = "terraform.tfstate"
  }}
}}

provider "azurerm" {{
  features {{
    key_vault {{
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }}
    cognitive_account {{
      purge_soft_delete_on_destroy = true
    }}
  }}
}}

provider "azuread" {{}}
'''

    def generate_resource_group(self) -> str:
        """Generate resource group configuration"""
        tags_tf = json.dumps(self.config.tags, indent=2)
        return f'''
resource "azurerm_resource_group" "main" {{
  name     = "{self.config.resource_group_name}"
  location = "{self.config.location}"
  tags     = {tags_tf}
}}
'''

    def generate_key_vault(self) -> str:
        """Generate Azure Key Vault configuration"""
        return f'''
data "azurerm_client_config" "current" {{}}

resource "azurerm_key_vault" "main" {{
  name                = "{self.config.application_name}-kv-${{random_id.main.hex}}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"
  
  purge_protection_enabled   = true
  soft_delete_retention_days = 7
  
  access_policy {{
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    
    key_permissions = [
      "Create", "Delete", "Get", "List", "Update", "Import", "Backup", "Restore"
    ]
    
    secret_permissions = [
      "Set", "Get", "Delete", "List", "Recover", "Backup", "Restore"
    ]
    
    certificate_permissions = [
      "Create", "Delete", "Get", "List", "Update", "Import"
    ]
  }}
  
  network_acls {{
    default_action = "Deny"
    bypass         = "AzureServices"
    
    # Add your IP for initial setup
    ip_rules = ["0.0.0.0/0"]  # Replace with your IP range
  }}
  
  tags = azurerm_resource_group.main.tags
}}

resource "random_id" "main" {{
  byte_length = 4
}}
'''

    def generate_openai_service(self, openai_config: OpenAIConfig) -> str:
        """Generate Azure OpenAI Service configuration"""
        deployments_tf = ""
        for deployment in openai_config.deployments:
            deployments_tf += f'''
resource "azurerm_cognitive_deployment" "{deployment['name'].replace('-', '_')}" {{
  name                 = "{deployment['name']}"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  
  model {{
    format  = "OpenAI"
    name    = "{deployment['model']}"
    version = "{deployment['version']}"
  }}
  
  scale {{
    type     = "Standard"
    capacity = {deployment['capacity']}
  }}
}}
'''
        
        return f'''
resource "azurerm_cognitive_account" "openai" {{
  name                = "{openai_config.account_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "{openai_config.kind}"
  sku_name            = "{openai_config.sku_name}"
  
  custom_question_answering_search_service_id = null
  
  identity {{
    type = "SystemAssigned"
  }}
  
  network_acls {{
    default_action = "Deny"
    
    virtual_network_rules {{
      subnet_id = azurerm_subnet.app.id
    }}
  }}
  
  tags = azurerm_resource_group.main.tags
}}

{deployments_tf}

# Store OpenAI endpoint and key in Key Vault
resource "azurerm_key_vault_secret" "openai_endpoint" {{
  name         = "azure-openai-endpoint"
  value        = azurerm_cognitive_account.openai.endpoint
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault.main]
}}

resource "azurerm_key_vault_secret" "openai_key" {{
  name         = "azure-openai-key"
  value        = azurerm_cognitive_account.openai.primary_access_key
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault.main]
}}
'''

    def generate_cosmos_db(self, cosmos_config: CosmosDBConfig) -> str:
        """Generate Azure Cosmos DB configuration"""
        return f'''
resource "azurerm_cosmosdb_account" "main" {{
  name                = "{cosmos_config.account_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"
  
  consistency_policy {{
    consistency_level = "{cosmos_config.consistency_level}"
  }}
  
  geo_location {{
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }}
  
  capabilities {{
    name = "EnableNoSQLVectorSearch"
  }}
  
  backup {{
    type                = "Periodic"
    interval_in_minutes = 240
    retention_in_hours  = 8
    storage_redundancy  = "Geo"
  }}
  
  tags = azurerm_resource_group.main.tags
}}

resource "azurerm_cosmosdb_sql_database" "main" {{
  name                = "{cosmos_config.database_name}"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
}}

resource "azurerm_cosmosdb_sql_container" "embeddings" {{
  name                = "{cosmos_config.container_name}"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  partition_key_path  = "/id"
  throughput          = {cosmos_config.throughput}
  
  indexing_policy {{
    indexing_mode = "consistent"
    
    included_path {{
      path = "/*"
    }}
    
    excluded_path {{
      path = "/embedding/*"
    }}
  }}
  
  vector_embedding_policy {{
    vector_embedding {{
      path                = "/embedding"
      data_type          = "float32"
      dimensions         = {cosmos_config.vector_policy['dimensions']}
      distance_function  = "{cosmos_config.vector_policy['distanceFunction']}"
    }}
  }}
}}

# Store Cosmos DB connection string in Key Vault
resource "azurerm_key_vault_secret" "cosmos_connection" {{
  name         = "cosmos-db-connection-string"
  value        = azurerm_cosmosdb_account.main.connection_strings[0]
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault.main]
}}
'''

    def generate_networking(self) -> str:
        """Generate networking configuration"""
        return f'''
resource "azurerm_virtual_network" "main" {{
  name                = "{self.config.application_name}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = azurerm_resource_group.main.tags
}}

resource "azurerm_subnet" "app" {{
  name                 = "app-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
  
  delegation {{
    name = "app-service-delegation"
    service_delegation {{
      name = "Microsoft.Web/serverFarms"
    }}
  }}
}}

resource "azurerm_subnet" "db" {{
  name                 = "db-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.2.0/24"]
  
  service_endpoints = ["Microsoft.AzureCosmosDB"]
}}

resource "azurerm_network_security_group" "app" {{
  name                = "{self.config.application_name}-app-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  security_rule {{
    name                       = "HTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}
  
  security_rule {{
    name                       = "HTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}
  
  tags = azurerm_resource_group.main.tags
}}

resource "azurerm_subnet_network_security_group_association" "app" {{
  subnet_id                 = azurerm_subnet.app.id
  network_security_group_id = azurerm_network_security_group.app.id
}}
'''

    def generate_app_service(self) -> str:
        """Generate Azure App Service configuration"""
        return f'''
resource "azurerm_service_plan" "main" {{
  name                = "{self.config.application_name}-asp"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "P1v3"
  
  tags = azurerm_resource_group.main.tags
}}

resource "azurerm_linux_web_app" "main" {{
  name                = "{self.config.application_name}-app"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id
  
  site_config {{
    always_on                         = true
    http2_enabled                     = true
    minimum_tls_version              = "1.2"
    scm_use_main_ip_restriction      = false
    use_32_bit_worker                = false
    websockets_enabled               = false
    
    application_stack {{
      python_version = "3.11"
    }}
    
    cors {{
      allowed_origins = ["https://{self.config.application_name}.azurewebsites.net"]
    }}
  }}
  
  app_settings = {{
    "AZURE_OPENAI_ENDPOINT"                = "@Microsoft.KeyVault(VaultName=${{azurerm_key_vault.main.name}};SecretName=azure-openai-endpoint)"
    "AZURE_OPENAI_API_KEY"                = "@Microsoft.KeyVault(VaultName=${{azurerm_key_vault.main.name}};SecretName=azure-openai-key)"
    "COSMOS_DB_CONNECTION_STRING"         = "@Microsoft.KeyVault(VaultName=${{azurerm_key_vault.main.name}};SecretName=cosmos-db-connection-string)"
    "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.main.connection_string
    "SCM_DO_BUILD_DURING_DEPLOYMENT"     = "true"
    "ENABLE_ORYX_BUILD"                   = "true"
    "WEBSITE_HTTPLOGGING_RETENTION_DAYS"  = "7"
  }}
  
  identity {{
    type = "SystemAssigned"
  }}
  
  virtual_network_subnet_id = azurerm_subnet.app.id
  
  logs {{
    detailed_error_messages = true
    failed_request_tracing  = true
    
    application_logs {{
      file_system_level = "Information"
    }}
    
    http_logs {{
      file_system {{
        retention_in_days = 7
        retention_in_mb   = 100
      }}
    }}
  }}
  
  tags = azurerm_resource_group.main.tags
}}

# Grant App Service access to Key Vault
resource "azurerm_key_vault_access_policy" "app_service" {{
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = azurerm_linux_web_app.main.identity[0].tenant_id
  object_id    = azurerm_linux_web_app.main.identity[0].principal_id
  
  secret_permissions = [
    "Get", "List"
  ]
  
  depends_on = [azurerm_linux_web_app.main]
}}
'''

    def generate_monitoring(self) -> str:
        """Generate monitoring and logging configuration"""
        return f'''
resource "azurerm_log_analytics_workspace" "main" {{
  name                = "{self.config.application_name}-law"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = azurerm_resource_group.main.tags
}}

resource "azurerm_application_insights" "main" {{
  name                = "{self.config.application_name}-ai"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  
  tags = azurerm_resource_group.main.tags
}}

# Diagnostic settings for App Service
resource "azurerm_monitor_diagnostic_setting" "app_service" {{
  name                       = "{self.config.application_name}-app-diagnostics"
  target_resource_id         = azurerm_linux_web_app.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  
  enabled_log {{
    category = "AppServiceHTTPLogs"
  }}
  
  enabled_log {{
    category = "AppServiceConsoleLogs"
  }}
  
  enabled_log {{
    category = "AppServiceAppLogs"
  }}
  
  metric {{
    category = "AllMetrics"
  }}
}}

# Diagnostic settings for Cosmos DB
resource "azurerm_monitor_diagnostic_setting" "cosmos_db" {{
  name                       = "{self.config.application_name}-cosmos-diagnostics"
  target_resource_id         = azurerm_cosmosdb_account.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  
  enabled_log {{
    category = "DataPlaneRequests"
  }}
  
  enabled_log {{
    category = "QueryRuntimeStatistics"
  }}
  
  metric {{
    category = "Requests"
  }}
}}
'''

    def generate_outputs(self) -> str:
        """Generate Terraform outputs"""
        return f'''
output "resource_group_name" {{
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}}

output "app_service_url" {{
  description = "URL of the deployed App Service"
  value       = "https://${{azurerm_linux_web_app.main.default_hostname}}"
}}

output "openai_endpoint" {{
  description = "Azure OpenAI endpoint"
  value       = azurerm_cognitive_account.openai.endpoint
  sensitive   = true
}}

output "cosmos_db_endpoint" {{
  description = "Cosmos DB endpoint"
  value       = azurerm_cosmosdb_account.main.endpoint
  sensitive   = true
}}

output "key_vault_uri" {{
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
  sensitive   = true
}}

output "application_insights_instrumentation_key" {{
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}}

output "application_insights_connection_string" {{
  description = "Application Insights connection string"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}}
'''

    def generate_complete_configuration(self, openai_config: OpenAIConfig, cosmos_config: CosmosDBConfig) -> str:
        """Generate complete Terraform configuration"""
        return f'''# =============================================================================
# Azure LangChain Production Infrastructure
# Generated by Azure Infrastructure Generator
# =============================================================================

{self.generate_providers()}

{self.generate_resource_group()}

{self.generate_key_vault()}

{self.generate_networking()}

{self.generate_openai_service(openai_config)}

{self.generate_cosmos_db(cosmos_config)}

{self.generate_app_service()}

{self.generate_monitoring()}

{self.generate_outputs()}
'''


# ============================================================================
# 3. ARM TEMPLATE GENERATOR
# ============================================================================

class ARMTemplateGenerator:
    """Generate ARM templates for Azure deployment"""
    
    def __init__(self, azure_config: AzureConfig):
        self.config = azure_config
    
    def generate_template(self, openai_config: OpenAIConfig, cosmos_config: CosmosDBConfig) -> Dict[str, Any]:
        """Generate complete ARM template"""
        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "metadata": {
                "description": "Azure LangChain Production Infrastructure",
                "author": "Azure Infrastructure Generator"
            },
            "parameters": {
                "applicationName": {
                    "type": "string",
                    "defaultValue": self.config.application_name,
                    "metadata": {"description": "Name of the application"}
                },
                "environment": {
                    "type": "string",
                    "defaultValue": self.config.environment,
                    "allowedValues": ["development", "staging", "production"],
                    "metadata": {"description": "Environment name"}
                },
                "location": {
                    "type": "string",
                    "defaultValue": self.config.location,
                    "metadata": {"description": "Azure region for deployment"}
                }
            },
            "variables": {
                "resourcePrefix": "[concat(parameters('applicationName'), '-', parameters('environment'))]",
                "tags": self.config.tags
            },
            "resources": [],
            "outputs": {}
        }
        
        # Add resources
        template["resources"].extend([
            self._generate_openai_resource(openai_config),
            self._generate_cosmos_resource(cosmos_config),
            self._generate_key_vault_resource(),
            self._generate_app_service_plan(),
            self._generate_web_app(),
            self._generate_application_insights()
        ])
        
        # Add outputs
        template["outputs"] = {
            "appServiceUrl": {
                "type": "string",
                "value": "[concat('https://', reference(resourceId('Microsoft.Web/sites', concat(variables('resourcePrefix'), '-app'))).defaultHostName)]"
            },
            "openaiEndpoint": {
                "type": "string",
                "value": "[reference(resourceId('Microsoft.CognitiveServices/accounts', concat(variables('resourcePrefix'), '-openai'))).endpoint]"
            }
        }
        
        return template
    
    def _generate_openai_resource(self, openai_config: OpenAIConfig) -> Dict[str, Any]:
        """Generate Azure OpenAI resource"""
        return {
            "type": "Microsoft.CognitiveServices/accounts",
            "apiVersion": "2023-05-01",
            "name": "[concat(variables('resourcePrefix'), '-openai')]",
            "location": "[parameters('location')]",
            "kind": openai_config.kind,
            "sku": {"name": openai_config.sku_name},
            "properties": {
                "customSubDomainName": "[concat(variables('resourcePrefix'), '-openai')]",
                "networkAcls": {
                    "defaultAction": "Deny",
                    "virtualNetworkRules": [],
                    "ipRules": []
                }
            },
            "tags": "[variables('tags')]"
        }
    
    def _generate_cosmos_resource(self, cosmos_config: CosmosDBConfig) -> Dict[str, Any]:
        """Generate Cosmos DB resource"""
        return {
            "type": "Microsoft.DocumentDB/databaseAccounts",
            "apiVersion": "2023-04-15",
            "name": "[concat(variables('resourcePrefix'), '-cosmos')]",
            "location": "[parameters('location')]",
            "kind": "GlobalDocumentDB",
            "properties": {
                "databaseAccountOfferType": "Standard",
                "consistencyPolicy": {
                    "defaultConsistencyLevel": cosmos_config.consistency_level
                },
                "locations": [
                    {
                        "locationName": "[parameters('location')]",
                        "failoverPriority": 0,
                        "isZoneRedundant": False
                    }
                ],
                "capabilities": [
                    {"name": "EnableNoSQLVectorSearch"}
                ],
                "backupPolicy": {
                    "type": "Periodic",
                    "periodicModeProperties": {
                        "backupIntervalInMinutes": 240,
                        "backupRetentionIntervalInHours": 8,
                        "backupStorageRedundancy": "Geo"
                    }
                }
            },
            "tags": "[variables('tags')]"
        }
    
    def _generate_key_vault_resource(self) -> Dict[str, Any]:
        """Generate Key Vault resource"""
        return {
            "type": "Microsoft.KeyVault/vaults",
            "apiVersion": "2023-02-01",
            "name": "[concat(variables('resourcePrefix'), '-kv-', uniqueString(resourceGroup().id))]",
            "location": "[parameters('location')]",
            "properties": {
                "tenantId": "[subscription().tenantId]",
                "sku": {"family": "A", "name": "premium"},
                "enabledForDeployment": True,
                "enabledForTemplateDeployment": True,
                "enabledForDiskEncryption": True,
                "enableSoftDelete": True,
                "softDeleteRetentionInDays": 7,
                "enablePurgeProtection": True,
                "accessPolicies": []
            },
            "tags": "[variables('tags')]"
        }
    
    def _generate_app_service_plan(self) -> Dict[str, Any]:
        """Generate App Service Plan"""
        return {
            "type": "Microsoft.Web/serverfarms",
            "apiVersion": "2022-03-01",
            "name": "[concat(variables('resourcePrefix'), '-asp')]",
            "location": "[parameters('location')]",
            "sku": {
                "name": "P1v3",
                "tier": "PremiumV3",
                "size": "P1v3",
                "family": "Pv3",
                "capacity": 1
            },
            "kind": "linux",
            "properties": {
                "reserved": True
            },
            "tags": "[variables('tags')]"
        }
    
    def _generate_web_app(self) -> Dict[str, Any]:
        """Generate Web App"""
        return {
            "type": "Microsoft.Web/sites",
            "apiVersion": "2022-03-01",
            "name": "[concat(variables('resourcePrefix'), '-app')]",
            "location": "[parameters('location')]",
            "dependsOn": [
                "[resourceId('Microsoft.Web/serverfarms', concat(variables('resourcePrefix'), '-asp'))]"
            ],
            "kind": "app,linux",
            "identity": {
                "type": "SystemAssigned"
            },
            "properties": {
                "serverFarmId": "[resourceId('Microsoft.Web/serverfarms', concat(variables('resourcePrefix'), '-asp'))]",
                "siteConfig": {
                    "linuxFxVersion": "PYTHON|3.11",
                    "alwaysOn": True,
                    "http20Enabled": True,
                    "minTlsVersion": "1.2",
                    "appSettings": [
                        {
                            "name": "APPLICATIONINSIGHTS_CONNECTION_STRING",
                            "value": "[reference(resourceId('Microsoft.Insights/components', concat(variables('resourcePrefix'), '-ai'))).ConnectionString]"
                        }
                    ]
                }
            },
            "tags": "[variables('tags')]"
        }
    
    def _generate_application_insights(self) -> Dict[str, Any]:
        """Generate Application Insights"""
        return {
            "type": "Microsoft.Insights/components",
            "apiVersion": "2020-02-02",
            "name": "[concat(variables('resourcePrefix'), '-ai')]",
            "location": "[parameters('location')]",
            "kind": "web",
            "properties": {
                "Application_Type": "web",
                "Request_Source": "rest"
            },
            "tags": "[variables('tags')]"
        }


# ============================================================================
# 4. BICEP TEMPLATE GENERATOR
# ============================================================================

class BicepTemplateGenerator:
    """Generate Bicep templates for modern Azure IaC"""
    
    def __init__(self, azure_config: AzureConfig):
        self.config = azure_config
    
    def generate_main_template(self) -> str:
        """Generate main Bicep template"""
        return f'''// Azure LangChain Production Infrastructure - Main Template
targetScope = 'resourceGroup'

@description('Application name')
param applicationName string = '{self.config.application_name}'

@description('Environment')
@allowed(['development', 'staging', 'production'])
param environment string = '{self.config.environment}'

@description('Azure region')
param location string = resourceGroup().location

@description('OpenAI deployment models')
param openaiDeployments array = [
  {{
    name: 'gpt-4'
    model: 'gpt-4'
    version: '0613'
    capacity: 10
  }}
  {{
    name: 'text-embedding-ada-002'
    model: 'text-embedding-ada-002'
    version: '2'
    capacity: 30
  }}
]

// Variables
var resourcePrefix = '${{applicationName}}-${{environment}}'
var tags = {{
  Environment: environment
  Application: applicationName
  ManagedBy: 'Bicep'
  CostCenter: 'AI-ML'
}}

// Modules
module openai 'modules/openai.bicep' = {{
  name: 'openai-deployment'
  params: {{
    name: '${{resourcePrefix}}-openai'
    location: location
    deployments: openaiDeployments
    tags: tags
  }}
}}

module cosmosdb 'modules/cosmosdb.bicep' = {{
  name: 'cosmosdb-deployment'
  params: {{
    accountName: '${{resourcePrefix}}-cosmos'
    location: location
    databaseName: 'langchain_db'
    containerName: 'embeddings'
    tags: tags
  }}
}}

module keyvault 'modules/keyvault.bicep' = {{
  name: 'keyvault-deployment'
  params: {{
    name: '${{resourcePrefix}}-kv-${{uniqueString(resourceGroup().id)}}'
    location: location
    tags: tags
  }}
}}

module appservice 'modules/appservice.bicep' = {{
  name: 'appservice-deployment'
  params: {{
    appName: '${{resourcePrefix}}-app'
    planName: '${{resourcePrefix}}-asp'
    location: location
    openaiEndpoint: openai.outputs.endpoint
    cosmosEndpoint: cosmosdb.outputs.endpoint
    keyVaultName: keyvault.outputs.name
    tags: tags
  }}
}}

module monitoring 'modules/monitoring.bicep' = {{
  name: 'monitoring-deployment'
  params: {{
    workspaceName: '${{resourcePrefix}}-law'
    appInsightsName: '${{resourcePrefix}}-ai'
    location: location
    tags: tags
  }}
}}

// Outputs
output appServiceUrl string = appservice.outputs.url
output openaiEndpoint string = openai.outputs.endpoint
output cosmosEndpoint string = cosmosdb.outputs.endpoint
output keyVaultUri string = keyvault.outputs.uri
'''

    def generate_openai_module(self) -> str:
        """Generate OpenAI Bicep module"""
        return '''// OpenAI Service Module
@description('OpenAI account name')
param name string

@description('Location')
param location string

@description('Model deployments')
param deployments array

@description('Resource tags')
param tags object = {}

resource openaiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: name
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: name
    networkAcls: {
      defaultAction: 'Deny'
      virtualNetworkRules: []
      ipRules: []
    }
  }
  tags: tags
}

resource openaiDeployments 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = [for deployment in deployments: {
  parent: openaiAccount
  name: deployment.name
  properties: {
    model: {
      format: 'OpenAI'
      name: deployment.model
      version: deployment.version
    }
    raiPolicyName: null
  }
  sku: {
    name: 'Standard'
    capacity: deployment.capacity
  }
}]

output endpoint string = openaiAccount.properties.endpoint
output id string = openaiAccount.id
output primaryKey string = openaiAccount.listKeys().key1
'''

    def generate_cosmos_module(self) -> str:
        """Generate Cosmos DB Bicep module"""
        return '''// Cosmos DB Module
@description('Cosmos DB account name')
param accountName string

@description('Location')
param location string

@description('Database name')
param databaseName string

@description('Container name')
param containerName string

@description('Resource tags')
param tags object = {}

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: accountName
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    capabilities: [
      {
        name: 'EnableNoSQLVectorSearch'
      }
    ]
  }
  tags: tags
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosAccount
  name: databaseName
  properties: {
    resource: {
      id: databaseName
    }
  }
}

resource container 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: database
  name: containerName
  properties: {
    resource: {
      id: containerName
      partitionKey: {
        paths: ['/id']
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/embedding/*'
          }
        ]
      }
      vectorEmbeddingPolicy: {
        vectorEmbeddings: [
          {
            path: '/embedding'
            dataType: 'float32'
            dimensions: 1536
            distanceFunction: 'cosine'
          }
        ]
      }
    }
    options: {
      throughput: 400
    }
  }
}

output endpoint string = cosmosAccount.properties.documentEndpoint
output id string = cosmosAccount.id
output connectionString string = cosmosAccount.listConnectionStrings().connectionStrings[0].connectionString
'''


# ============================================================================
# 5. AZURE CLI AUTOMATION SCRIPTS
# ============================================================================

class AzureCLIGenerator:
    """Generate Azure CLI scripts for deployment automation"""
    
    def __init__(self, azure_config: AzureConfig):
        self.config = azure_config
    
    def generate_deployment_script(self) -> str:
        """Generate complete deployment script"""
        return f'''#!/bin/bash
# Azure LangChain Production Deployment Script
# Generated by Azure Infrastructure Generator

set -e  # Exit on any error

# Configuration
SUBSCRIPTION_ID="{self.config.subscription_id}"
RESOURCE_GROUP="{self.config.resource_group_name}"
LOCATION="{self.config.location}"
APP_NAME="{self.config.application_name}"
ENVIRONMENT="{self.config.environment}"

echo "🚀 Starting Azure LangChain Production Deployment"
echo "================================================="

# Check if logged in to Azure
echo "📋 Checking Azure CLI login..."
if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure CLI"
    echo "Please run: az login"
    exit 1
fi

# Set subscription
echo "🔧 Setting Azure subscription..."
az account set --subscription "$SUBSCRIPTION_ID"

# Create resource group
echo "📦 Creating resource group..."
az group create \\
    --name "$RESOURCE_GROUP" \\
    --location "$LOCATION" \\
    --tags Environment="$ENVIRONMENT" Application="$APP_NAME"

# Deploy infrastructure using Bicep
echo "🏗️ Deploying infrastructure..."
az deployment group create \\
    --resource-group "$RESOURCE_GROUP" \\
    --template-file main.bicep \\
    --parameters applicationName="$APP_NAME" environment="$ENVIRONMENT"

# Get deployment outputs
echo "📋 Getting deployment outputs..."
OUTPUTS=$(az deployment group show \\
    --resource-group "$RESOURCE_GROUP" \\
    --name main \\
    --query properties.outputs)

APP_URL=$(echo "$OUTPUTS" | jq -r '.appServiceUrl.value')
OPENAI_ENDPOINT=$(echo "$OUTPUTS" | jq -r '.openaiEndpoint.value')

echo "✅ Deployment completed successfully!"
echo "🌐 App Service URL: $APP_URL"
echo "🤖 OpenAI Endpoint: $OPENAI_ENDPOINT"

# Configure application settings
echo "⚙️ Configuring application settings..."
./configure-app.sh "$RESOURCE_GROUP" "$APP_NAME-$ENVIRONMENT-app"

echo "🎉 Azure LangChain deployment completed!"
'''

    def generate_app_config_script(self) -> str:
        """Generate application configuration script"""
        return f'''#!/bin/bash
# Application Configuration Script

RESOURCE_GROUP=$1
APP_NAME=$2

if [ -z "$RESOURCE_GROUP" ] || [ -z "$APP_NAME" ]; then
    echo "Usage: $0 <resource-group> <app-name>"
    exit 1
fi

echo "⚙️ Configuring application settings for $APP_NAME..."

# Get Key Vault name
KEY_VAULT=$(az keyvault list \\
    --resource-group "$RESOURCE_GROUP" \\
    --query "[0].name" \\
    --output tsv)

# Configure app settings with Key Vault references
az webapp config appsettings set \\
    --resource-group "$RESOURCE_GROUP" \\
    --name "$APP_NAME" \\
    --settings \\
        AZURE_OPENAI_ENDPOINT="@Microsoft.KeyVault(VaultName=$KEY_VAULT;SecretName=azure-openai-endpoint)" \\
        AZURE_OPENAI_API_KEY="@Microsoft.KeyVault(VaultName=$KEY_VAULT;SecretName=azure-openai-key)" \\
        COSMOS_DB_CONNECTION_STRING="@Microsoft.KeyVault(VaultName=$KEY_VAULT;SecretName=cosmos-db-connection-string)" \\
        PYTHON_VERSION="3.11" \\
        SCM_DO_BUILD_DURING_DEPLOYMENT="true" \\
        ENABLE_ORYX_BUILD="true"

# Grant App Service access to Key Vault
echo "🔐 Granting Key Vault access..."
APP_IDENTITY=$(az webapp identity show \\
    --resource-group "$RESOURCE_GROUP" \\
    --name "$APP_NAME" \\
    --query principalId \\
    --output tsv)

az keyvault set-policy \\
    --name "$KEY_VAULT" \\
    --object-id "$APP_IDENTITY" \\
    --secret-permissions get list

echo "✅ Application configuration completed!"
'''

    def generate_monitoring_script(self) -> str:
        """Generate monitoring setup script"""
        return f'''#!/bin/bash
# Monitoring and Alerting Setup Script

RESOURCE_GROUP=$1
APP_NAME=$2

if [ -z "$RESOURCE_GROUP" ] || [ -z "$APP_NAME" ]; then
    echo "Usage: $0 <resource-group> <app-name>"
    exit 1
fi

echo "📊 Setting up monitoring and alerting..."

# Create action group for notifications
az monitor action-group create \\
    --name "$APP_NAME-alerts" \\
    --resource-group "$RESOURCE_GROUP" \\
    --short-name "AI-Alerts" \\
    --email-receivers name="DevOps Team" email="devops@company.com"

# Create CPU usage alert
az monitor metrics alert create \\
    --name "$APP_NAME-high-cpu" \\
    --resource-group "$RESOURCE_GROUP" \\
    --description "High CPU usage alert" \\
    --condition "avg Percentage CPU > 80" \\
    --window-size 5m \\
    --evaluation-frequency 1m \\
    --action "$APP_NAME-alerts" \\
    --target-resource-type "Microsoft.Web/sites" \\
    --target-resource-id "/subscriptions/{self.config.subscription_id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$APP_NAME"

# Create memory usage alert
az monitor metrics alert create \\
    --name "$APP_NAME-high-memory" \\
    --resource-group "$RESOURCE_GROUP" \\
    --description "High memory usage alert" \\
    --condition "avg MemoryPercentage > 85" \\
    --window-size 5m \\
    --evaluation-frequency 1m \\
    --action "$APP_NAME-alerts" \\
    --target-resource-type "Microsoft.Web/sites" \\
    --target-resource-id "/subscriptions/{self.config.subscription_id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$APP_NAME"

# Create response time alert
az monitor metrics alert create \\
    --name "$APP_NAME-slow-response" \\
    --resource-group "$RESOURCE_GROUP" \\
    --description "Slow response time alert" \\
    --condition "avg AverageResponseTime > 5000" \\
    --window-size 5m \\
    --evaluation-frequency 1m \\
    --action "$APP_NAME-alerts" \\
    --target-resource-type "Microsoft.Web/sites" \\
    --target-resource-id "/subscriptions/{self.config.subscription_id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$APP_NAME"

echo "✅ Monitoring and alerting setup completed!"
'''


# ============================================================================
# 6. DEMONSTRATION AND USAGE
# ============================================================================

def demonstrate_infrastructure_generation():
    """Demonstrate infrastructure code generation"""
    print("🏗️ Azure Infrastructure as Code Generation Demo")
    print("=" * 60)
    
    # Create configuration
    azure_config = AzureConfig(
        subscription_id="your-subscription-id",
        resource_group_name="langchain-prod-rg",
        location="East US",
        environment="production",
        application_name="langchain-ai-app"
    )
    
    openai_config = OpenAIConfig(
        account_name=f"{azure_config.application_name}-openai"
    )
    
    cosmos_config = CosmosDBConfig(
        account_name=f"{azure_config.application_name}-cosmos"
    )
    
    # Generate Terraform configuration
    print("📋 Generating Terraform Configuration...")
    terraform_gen = TerraformConfigGenerator(azure_config)
    terraform_config = terraform_gen.generate_complete_configuration(openai_config, cosmos_config)
    
    # Save Terraform files
    os.makedirs("terraform", exist_ok=True)
    with open("terraform/main.tf", "w") as f:
        f.write(terraform_config)
    
    print("✅ Terraform configuration saved to terraform/main.tf")
    
    # Generate ARM template
    print("📋 Generating ARM Template...")
    arm_gen = ARMTemplateGenerator(azure_config)
    arm_template = arm_gen.generate_template(openai_config, cosmos_config)
    
    # Save ARM template
    os.makedirs("arm", exist_ok=True)
    with open("arm/azuredeploy.json", "w") as f:
        json.dump(arm_template, f, indent=2)
    
    print("✅ ARM template saved to arm/azuredeploy.json")
    
    # Generate Bicep templates
    print("📋 Generating Bicep Templates...")
    bicep_gen = BicepTemplateGenerator(azure_config)
    
    os.makedirs("bicep/modules", exist_ok=True)
    
    # Save main Bicep template
    with open("bicep/main.bicep", "w") as f:
        f.write(bicep_gen.generate_main_template())
    
    # Save Bicep modules
    with open("bicep/modules/openai.bicep", "w") as f:
        f.write(bicep_gen.generate_openai_module())
    
    with open("bicep/modules/cosmosdb.bicep", "w") as f:
        f.write(bicep_gen.generate_cosmos_module())
    
    print("✅ Bicep templates saved to bicep/ directory")
    
    # Generate Azure CLI scripts
    print("📋 Generating Azure CLI Scripts...")
    cli_gen = AzureCLIGenerator(azure_config)
    
    os.makedirs("scripts", exist_ok=True)
    
    with open("scripts/deploy.sh", "w") as f:
        f.write(cli_gen.generate_deployment_script())
    
    with open("scripts/configure-app.sh", "w") as f:
        f.write(cli_gen.generate_app_config_script())
    
    with open("scripts/setup-monitoring.sh", "w") as f:
        f.write(cli_gen.generate_monitoring_script())
    
    # Make scripts executable
    os.chmod("scripts/deploy.sh", 0o755)
    os.chmod("scripts/configure-app.sh", 0o755)
    os.chmod("scripts/setup-monitoring.sh", 0o755)
    
    print("✅ Azure CLI scripts saved to scripts/ directory")
    
    print("\n🎉 Infrastructure code generation completed!")
    print("\n📁 Generated Files:")
    print("├── terraform/")
    print("│   └── main.tf")
    print("├── arm/")
    print("│   └── azuredeploy.json")
    print("├── bicep/")
    print("│   ├── main.bicep")
    print("│   └── modules/")
    print("│       ├── openai.bicep")
    print("│       └── cosmosdb.bicep")
    print("└── scripts/")
    print("    ├── deploy.sh")
    print("    ├── configure-app.sh")
    print("    └── setup-monitoring.sh")
    
    print("\n🚀 Next Steps:")
    print("1. Review and customize the generated configurations")
    print("2. Update subscription IDs and resource names")
    print("3. Run ./scripts/deploy.sh to deploy infrastructure")
    print("4. Configure application settings with ./scripts/configure-app.sh")
    print("5. Set up monitoring with ./scripts/setup-monitoring.sh")


if __name__ == "__main__":
    demonstrate_infrastructure_generation()
