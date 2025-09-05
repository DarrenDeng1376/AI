"""
Azure Security and Compliance for LangChain Applications

This module provides comprehensive security and compliance solutions for LangChain
applications deployed on Azure. It includes Azure Key Vault integration, 
managed identities, network security, data protection, compliance frameworks,
and security monitoring.

Key features:
1. Azure Key Vault for secrets management
2. Managed Identity and RBAC configuration
3. Network security with Private Endpoints and VNets
4. Data encryption and protection
5. Compliance frameworks (SOC2, HIPAA, PCI DSS)
6. Security monitoring and threat detection
7. Data residency and sovereignty
8. Audit logging and compliance reporting
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.keyvault.keys import KeyClient
from azure.mgmt.keyvault import KeyVaultManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.authorization import AuthorizationManagementClient
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ============================================================================
# 1. AZURE KEY VAULT CONFIGURATION
# ============================================================================

@dataclass
class KeyVaultConfig:
    """Azure Key Vault configuration"""
    vault_name: str
    resource_group: str
    location: str = "East US"
    sku: str = "premium"
    tenant_id: str = ""
    enable_soft_delete: bool = True
    soft_delete_retention: int = 90
    enable_purge_protection: bool = True
    enable_rbac: bool = True
    network_access: str = "private"  # private, public, selected
    allowed_ips: List[str] = None
    virtual_networks: List[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {
                "Environment": "production",
                "Application": "langchain-ai",
                "DataClassification": "confidential",
                "ComplianceFramework": "SOC2"
            }
        if self.allowed_ips is None:
            self.allowed_ips = []
        if self.virtual_networks is None:
            self.virtual_networks = []


class AzureKeyVaultManager:
    """Manage Azure Key Vault operations"""
    
    def __init__(self, vault_url: str, credential=None):
        self.vault_url = vault_url
        self.credential = credential or DefaultAzureCredential()
        self.secret_client = SecretClient(vault_url=vault_url, credential=self.credential)
        self.key_client = KeyClient(vault_url=vault_url, credential=self.credential)
        self.logger = logging.getLogger(__name__)
    
    async def store_secret(self, name: str, value: str, tags: Dict[str, str] = None) -> str:
        """Store a secret in Key Vault"""
        try:
            if tags is None:
                tags = {
                    "CreatedBy": "LangChain",
                    "CreatedAt": datetime.utcnow().isoformat(),
                    "DataClassification": "sensitive"
                }
            
            secret = self.secret_client.set_secret(name, value, tags=tags)
            self.logger.info(f"Secret '{name}' stored successfully")
            return secret.id
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{name}': {e}")
            raise
    
    async def get_secret(self, name: str) -> str:
        """Retrieve a secret from Key Vault"""
        try:
            secret = self.secret_client.get_secret(name)
            self.logger.info(f"Secret '{name}' retrieved successfully")
            return secret.value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{name}': {e}")
            raise
    
    async def create_encryption_key(self, key_name: str, key_type: str = "RSA") -> str:
        """Create an encryption key in Key Vault"""
        try:
            key = self.key_client.create_rsa_key(key_name, size=2048) if key_type == "RSA" else self.key_client.create_ec_key(key_name)
            self.logger.info(f"Encryption key '{key_name}' created successfully")
            return key.id
            
        except Exception as e:
            self.logger.error(f"Failed to create encryption key '{key_name}': {e}")
            raise
    
    async def rotate_secret(self, name: str, new_value: str) -> str:
        """Rotate a secret in Key Vault"""
        try:
            # Create new version
            new_secret = self.secret_client.set_secret(name, new_value)
            
            # Log rotation event
            self.logger.info(f"Secret '{name}' rotated successfully")
            
            return new_secret.id
            
        except Exception as e:
            self.logger.error(f"Failed to rotate secret '{name}': {e}")
            raise


# ============================================================================
# 2. MANAGED IDENTITY AND RBAC
# ============================================================================

@dataclass
class ManagedIdentityConfig:
    """Managed Identity configuration"""
    identity_name: str
    resource_group: str
    location: str = "East US"
    type: str = "UserAssigned"  # UserAssigned or SystemAssigned
    role_assignments: List[Dict[str, str]] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.role_assignments is None:
            self.role_assignments = [
                {
                    "role": "Key Vault Secrets User",
                    "scope": "/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.KeyVault/vaults/{vault_name}"
                },
                {
                    "role": "Cognitive Services User",
                    "scope": "/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{openai_account}"
                }
            ]
        if self.tags is None:
            self.tags = {
                "Purpose": "LangChain-Application",
                "Environment": "production"
            }


class RBACManager:
    """Manage Role-Based Access Control"""
    
    def __init__(self, subscription_id: str, credential=None):
        self.subscription_id = subscription_id
        self.credential = credential or DefaultAzureCredential()
        self.auth_client = AuthorizationManagementClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_role_assignment(self, 
                                principal_id: str, 
                                role_definition_id: str, 
                                scope: str) -> Dict[str, Any]:
        """Generate role assignment ARM template"""
        return {
            "type": "Microsoft.Authorization/roleAssignments",
            "apiVersion": "2022-04-01",
            "name": "[guid(parameters('principalId'), parameters('roleDefinitionId'), resourceGroup().id)]",
            "properties": {
                "principalId": principal_id,
                "roleDefinitionId": role_definition_id,
                "scope": scope,
                "principalType": "ServicePrincipal"
            }
        }
    
    def generate_custom_role_definition(self) -> Dict[str, Any]:
        """Generate custom role for LangChain applications"""
        return {
            "type": "Microsoft.Authorization/roleDefinitions",
            "apiVersion": "2022-04-01",
            "name": "[guid('LangChain-Application-Role')]",
            "properties": {
                "roleName": "LangChain Application Role",
                "description": "Custom role for LangChain applications with minimal required permissions",
                "type": "CustomRole",
                "permissions": [
                    {
                        "actions": [
                            "Microsoft.KeyVault/vaults/secrets/read",
                            "Microsoft.CognitiveServices/accounts/read",
                            "Microsoft.CognitiveServices/accounts/listKeys/action",
                            "Microsoft.DocumentDB/databaseAccounts/read",
                            "Microsoft.DocumentDB/databaseAccounts/readonlykeys/action",
                            "Microsoft.Storage/storageAccounts/read",
                            "Microsoft.Storage/storageAccounts/listKeys/action"
                        ],
                        "notActions": [],
                        "dataActions": [
                            "Microsoft.KeyVault/vaults/secrets/getSecret/action",
                            "Microsoft.CognitiveServices/accounts/OpenAI/chat/completions/action",
                            "Microsoft.CognitiveServices/accounts/OpenAI/embeddings/action",
                            "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/read",
                            "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/create",
                            "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/upsert"
                        ],
                        "notDataActions": []
                    }
                ],
                "assignableScopes": [
                    f"/subscriptions/{self.subscription_id}"
                ]
            }
        }


# ============================================================================
# 3. NETWORK SECURITY
# ============================================================================

@dataclass
class NetworkSecurityConfig:
    """Network security configuration"""
    vnet_name: str
    resource_group: str
    location: str = "East US"
    address_space: str = "10.0.0.0/16"
    subnets: List[Dict[str, str]] = None
    enable_ddos_protection: bool = True
    enable_private_endpoints: bool = True
    nsg_rules: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.subnets is None:
            self.subnets = [
                {
                    "name": "langchain-app-subnet",
                    "address_prefix": "10.0.1.0/24",
                    "service_endpoints": ["Microsoft.KeyVault", "Microsoft.CognitiveServices", "Microsoft.AzureCosmosDB"]
                },
                {
                    "name": "private-endpoints-subnet",
                    "address_prefix": "10.0.2.0/24",
                    "private_endpoint_network_policies": "Disabled"
                }
            ]
        if self.nsg_rules is None:
            self.nsg_rules = [
                {
                    "name": "AllowHTTPS",
                    "priority": 1000,
                    "direction": "Inbound",
                    "access": "Allow",
                    "protocol": "Tcp",
                    "sourcePortRange": "*",
                    "destinationPortRange": "443",
                    "sourceAddressPrefix": "*",
                    "destinationAddressPrefix": "*"
                },
                {
                    "name": "DenyAllInbound",
                    "priority": 4096,
                    "direction": "Inbound",
                    "access": "Deny",
                    "protocol": "*",
                    "sourcePortRange": "*",
                    "destinationPortRange": "*",
                    "sourceAddressPrefix": "*",
                    "destinationAddressPrefix": "*"
                }
            ]


class NetworkSecurityManager:
    """Manage network security configurations"""
    
    def generate_vnet_template(self, config: NetworkSecurityConfig) -> Dict[str, Any]:
        """Generate Virtual Network ARM template"""
        return {
            "type": "Microsoft.Network/virtualNetworks",
            "apiVersion": "2023-05-01",
            "name": config.vnet_name,
            "location": config.location,
            "properties": {
                "addressSpace": {
                    "addressPrefixes": [config.address_space]
                },
                "subnets": [
                    {
                        "name": subnet["name"],
                        "properties": {
                            "addressPrefix": subnet["address_prefix"],
                            "serviceEndpoints": [
                                {"service": endpoint} for endpoint in subnet.get("service_endpoints", [])
                            ],
                            "privateEndpointNetworkPolicies": subnet.get("private_endpoint_network_policies", "Enabled"),
                            "privateLinkServiceNetworkPolicies": "Enabled"
                        }
                    } for subnet in config.subnets
                ],
                "enableDdosProtection": config.enable_ddos_protection
            }
        }
    
    def generate_nsg_template(self, config: NetworkSecurityConfig) -> Dict[str, Any]:
        """Generate Network Security Group ARM template"""
        return {
            "type": "Microsoft.Network/networkSecurityGroups",
            "apiVersion": "2023-05-01",
            "name": f"{config.vnet_name}-nsg",
            "location": config.location,
            "properties": {
                "securityRules": [
                    {
                        "name": rule["name"],
                        "properties": {
                            "priority": rule["priority"],
                            "direction": rule["direction"],
                            "access": rule["access"],
                            "protocol": rule["protocol"],
                            "sourcePortRange": rule["sourcePortRange"],
                            "destinationPortRange": rule["destinationPortRange"],
                            "sourceAddressPrefix": rule["sourceAddressPrefix"],
                            "destinationAddressPrefix": rule["destinationAddressPrefix"]
                        }
                    } for rule in config.nsg_rules
                ]
            }
        }
    
    def generate_private_endpoint_template(self, 
                                         service_name: str, 
                                         service_type: str,
                                         subnet_id: str) -> Dict[str, Any]:
        """Generate Private Endpoint ARM template"""
        return {
            "type": "Microsoft.Network/privateEndpoints",
            "apiVersion": "2023-05-01",
            "name": f"{service_name}-private-endpoint",
            "location": "[resourceGroup().location]",
            "properties": {
                "subnet": {
                    "id": subnet_id
                },
                "privateLinkServiceConnections": [
                    {
                        "name": f"{service_name}-connection",
                        "properties": {
                            "privateLinkServiceId": f"[resourceId('Microsoft.{service_type}', '{service_name}')]",
                            "groupIds": self._get_group_ids_for_service(service_type)
                        }
                    }
                ]
            }
        }
    
    def _get_group_ids_for_service(self, service_type: str) -> List[str]:
        """Get group IDs for different service types"""
        group_id_mapping = {
            "KeyVault/vaults": ["vault"],
            "CognitiveServices/accounts": ["account"],
            "DocumentDB/databaseAccounts": ["Sql"],
            "Storage/storageAccounts": ["blob"]
        }
        return group_id_mapping.get(service_type, ["default"])


# ============================================================================
# 4. DATA ENCRYPTION AND PROTECTION
# ============================================================================

class DataProtectionManager:
    """Manage data encryption and protection"""
    
    def __init__(self, key_vault_manager: AzureKeyVaultManager):
        self.key_vault = key_vault_manager
        self.logger = logging.getLogger(__name__)
    
    async def encrypt_sensitive_data(self, data: str, key_name: str = None) -> str:
        """Encrypt sensitive data using Azure Key Vault key"""
        try:
            if key_name:
                # Use Azure Key Vault key for encryption
                # This is a simplified example - in production, use proper key vault encryption
                key = await self.key_vault.get_secret(key_name)
                fernet = Fernet(key.encode())
                encrypted_data = fernet.encrypt(data.encode())
                return base64.b64encode(encrypted_data).decode()
            else:
                # Generate a new key for this data
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(data.encode()))
                fernet = Fernet(key)
                encrypted_data = fernet.encrypt(data.encode())
                return base64.b64encode(salt + encrypted_data).decode()
                
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}")
            raise
    
    async def decrypt_sensitive_data(self, encrypted_data: str, key_name: str = None) -> str:
        """Decrypt sensitive data using Azure Key Vault key"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            if key_name:
                # Use Azure Key Vault key for decryption
                key = await self.key_vault.get_secret(key_name)
                fernet = Fernet(key.encode())
                decrypted_data = fernet.decrypt(encrypted_bytes)
                return decrypted_data.decode()
            else:
                # Extract salt and decrypt
                salt = encrypted_bytes[:16]
                encrypted_content = encrypted_bytes[16:]
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                # Note: This is simplified - you'd need to store/retrieve the original password securely
                key = base64.urlsafe_b64encode(kdf.derive(b"default_password"))
                fernet = Fernet(key)
                decrypted_data = fernet.decrypt(encrypted_content)
                return decrypted_data.decode()
                
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def generate_data_classification_policy(self) -> Dict[str, Any]:
        """Generate data classification policy"""
        return {
            "version": "1.0",
            "dataClassification": {
                "public": {
                    "description": "Data that can be shared publicly without risk",
                    "encryptionRequired": False,
                    "retentionPeriod": "7 years",
                    "accessControls": "None"
                },
                "internal": {
                    "description": "Data for internal use within the organization",
                    "encryptionRequired": True,
                    "retentionPeriod": "5 years",
                    "accessControls": "Employee access only"
                },
                "confidential": {
                    "description": "Sensitive data requiring protection",
                    "encryptionRequired": True,
                    "retentionPeriod": "3 years",
                    "accessControls": "Role-based access",
                    "additionalControls": ["DLP", "Audit logging"]
                },
                "restricted": {
                    "description": "Highly sensitive data with legal/regulatory requirements",
                    "encryptionRequired": True,
                    "retentionPeriod": "10 years",
                    "accessControls": "Privileged access only",
                    "additionalControls": ["DLP", "Audit logging", "Multi-factor authentication"]
                }
            }
        }


# ============================================================================
# 5. COMPLIANCE FRAMEWORKS
# ============================================================================

class ComplianceManager:
    """Manage compliance frameworks and requirements"""
    
    def generate_soc2_controls(self) -> Dict[str, Any]:
        """Generate SOC 2 compliance controls"""
        return {
            "framework": "SOC 2 Type II",
            "controls": {
                "CC6.1": {
                    "name": "Logical and Physical Access Controls",
                    "description": "Restrict logical and physical access to systems",
                    "implementation": [
                        "Azure RBAC for access control",
                        "Multi-factor authentication",
                        "Privileged access management",
                        "Regular access reviews"
                    ],
                    "evidence": [
                        "Access control lists",
                        "Authentication logs",
                        "Access review reports"
                    ]
                },
                "CC6.2": {
                    "name": "Data Transmission Controls",
                    "description": "Protect data during transmission",
                    "implementation": [
                        "TLS 1.2+ for all connections",
                        "VPN for internal communications",
                        "Certificate management",
                        "Network segmentation"
                    ],
                    "evidence": [
                        "SSL/TLS configurations",
                        "Network architecture diagrams",
                        "Certificate inventory"
                    ]
                },
                "CC6.3": {
                    "name": "Data Processing Controls", 
                    "description": "Protect data during processing",
                    "implementation": [
                        "Encryption at rest using Azure Key Vault",
                        "Memory protection mechanisms",
                        "Secure coding practices",
                        "Data loss prevention"
                    ],
                    "evidence": [
                        "Encryption configurations",
                        "Code review reports",
                        "DLP policies"
                    ]
                }
            }
        }
    
    def generate_hipaa_controls(self) -> Dict[str, Any]:
        """Generate HIPAA compliance controls"""
        return {
            "framework": "HIPAA",
            "controls": {
                "164.312(a)(1)": {
                    "name": "Access Control",
                    "description": "Assign unique user identification, automatic logoff, encryption and decryption",
                    "implementation": [
                        "Unique user IDs via Azure AD",
                        "Automatic session timeout",
                        "Data encryption with Azure Key Vault",
                        "Role-based access control"
                    ]
                },
                "164.312(b)": {
                    "name": "Audit Controls",
                    "description": "Record and examine system activity",
                    "implementation": [
                        "Azure Monitor audit logging",
                        "Application Insights telemetry",
                        "Security Center monitoring",
                        "Log Analytics workspace"
                    ]
                },
                "164.312(c)(1)": {
                    "name": "Integrity",
                    "description": "Protect PHI from improper alteration or destruction",
                    "implementation": [
                        "Immutable storage for critical data",
                        "Digital signatures for data integrity",
                        "Version control for all changes",
                        "Backup and recovery procedures"
                    ]
                }
            }
        }
    
    def generate_compliance_monitoring_queries(self) -> Dict[str, str]:
        """Generate KQL queries for compliance monitoring"""
        return {
            "access_violations": """
                AuditLogs
                | where TimeGenerated >= ago(24h)
                | where Result == "failure"
                | where Category == "Authentication" or Category == "Authorization"
                | summarize FailureCount = count() by UserPrincipalName, bin(TimeGenerated, 1h)
                | where FailureCount > 5
            """,
            "privileged_access": """
                AuditLogs
                | where TimeGenerated >= ago(24h)
                | where Category == "RoleManagement"
                | where OperationName contains "Add member to role"
                | project TimeGenerated, UserPrincipalName, OperationName, TargetUserPrincipalName
            """,
            "data_access_patterns": """
                AppTraces
                | where TimeGenerated >= ago(24h)
                | where Message contains "data_access"
                | extend UserId = tostring(customDimensions.user_id)
                | extend DataType = tostring(customDimensions.data_type)
                | summarize AccessCount = count() by UserId, DataType, bin(TimeGenerated, 1h)
                | where AccessCount > 100
            """,
            "encryption_key_usage": """
                KeyVaultLogs
                | where TimeGenerated >= ago(24h)
                | where OperationName in ("SecretGet", "KeyDecrypt", "KeyEncrypt")
                | summarize OperationCount = count() by OperationName, bin(TimeGenerated, 1h)
            """
        }


# ============================================================================
# 6. SECURITY MONITORING
# ============================================================================

class SecurityMonitoringManager:
    """Manage security monitoring and threat detection"""
    
    def generate_security_alerts(self) -> Dict[str, Any]:
        """Generate security alert rules"""
        return {
            "anomalous_access_patterns": {
                "name": "Anomalous Access Patterns",
                "description": "Detect unusual access patterns to sensitive data",
                "query": """
                    AppTraces
                    | where TimeGenerated >= ago(1h)
                    | extend UserId = tostring(customDimensions.user_id)
                    | extend DataType = tostring(customDimensions.data_type)
                    | summarize AccessCount = count() by UserId, DataType
                    | where AccessCount > 50
                """,
                "threshold": 1,
                "severity": "High"
            },
            "failed_authentication_attempts": {
                "name": "Multiple Failed Authentication Attempts",
                "description": "Detect potential brute force attacks",
                "query": """
                    SigninLogs
                    | where TimeGenerated >= ago(10m)
                    | where ResultType != 0
                    | summarize FailureCount = count() by UserPrincipalName, IPAddress
                    | where FailureCount >= 5
                """,
                "threshold": 1,
                "severity": "Medium"
            },
            "unauthorized_key_vault_access": {
                "name": "Unauthorized Key Vault Access",
                "description": "Detect unauthorized access attempts to Key Vault",
                "query": """
                    KeyVaultLogs
                    | where TimeGenerated >= ago(5m)
                    | where ResultType == "Forbidden" or ResultType == "Unauthorized"
                    | summarize AttemptCount = count() by CallerIPAddress, OperationName
                    | where AttemptCount >= 3
                """,
                "threshold": 1,
                "severity": "High"
            }
        }
    
    def generate_security_dashboard(self) -> Dict[str, Any]:
        """Generate security monitoring dashboard"""
        return {
            "name": "LangChain Security Dashboard",
            "version": "1.0",
            "panels": [
                {
                    "title": "Authentication Events",
                    "type": "timechart",
                    "query": """
                        SigninLogs
                        | where TimeGenerated >= ago(24h)
                        | summarize SuccessCount = countif(ResultType == 0), 
                                   FailureCount = countif(ResultType != 0) 
                          by bin(TimeGenerated, 1h)
                        | render timechart
                    """
                },
                {
                    "title": "Key Vault Operations",
                    "type": "piechart", 
                    "query": """
                        KeyVaultLogs
                        | where TimeGenerated >= ago(24h)
                        | summarize Count = count() by OperationName
                        | render piechart
                    """
                },
                {
                    "title": "Data Access by Classification",
                    "type": "barchart",
                    "query": """
                        AppTraces
                        | where TimeGenerated >= ago(24h)
                        | extend DataClassification = tostring(customDimensions.data_classification)
                        | summarize AccessCount = count() by DataClassification
                        | render barchart
                    """
                }
            ]
        }


# ============================================================================
# 7. DEMONSTRATION AND USAGE
# ============================================================================

async def demonstrate_security_compliance():
    """Demonstrate security and compliance setup"""
    print("🔒 Azure Security and Compliance Demo")
    print("=" * 60)
    
    # Key Vault configuration
    kv_config = KeyVaultConfig(
        vault_name="langchain-kv-prod",
        resource_group="langchain-rg",
        tenant_id="your-tenant-id"
    )
    
    print("🔑 Setting up Azure Key Vault...")
    # Note: In real usage, you'd initialize with actual credentials
    # kv_manager = AzureKeyVaultManager(f"https://{kv_config.vault_name}.vault.azure.net/")
    
    # Managed Identity configuration
    mi_config = ManagedIdentityConfig(
        identity_name="langchain-identity",
        resource_group="langchain-rg"
    )
    
    print("👤 Configuring Managed Identity and RBAC...")
    rbac_manager = RBACManager("your-subscription-id")
    
    # Network security configuration
    network_config = NetworkSecurityConfig(
        vnet_name="langchain-vnet",
        resource_group="langchain-rg"
    )
    
    print("🌐 Setting up Network Security...")
    network_manager = NetworkSecurityManager()
    
    # Generate ARM templates
    print("📋 Generating ARM templates...")
    os.makedirs("security/templates", exist_ok=True)
    
    # Key Vault template
    kv_template = {
        "type": "Microsoft.KeyVault/vaults",
        "apiVersion": "2023-02-01",
        "name": kv_config.vault_name,
        "location": kv_config.location,
        "properties": {
            "tenantId": kv_config.tenant_id,
            "sku": {"family": "A", "name": kv_config.sku},
            "enableSoftDelete": kv_config.enable_soft_delete,
            "softDeleteRetentionInDays": kv_config.soft_delete_retention,
            "enablePurgeProtection": kv_config.enable_purge_protection,
            "enableRbacAuthorization": kv_config.enable_rbac,
            "networkAcls": {
                "defaultAction": "Deny" if kv_config.network_access == "private" else "Allow",
                "ipRules": [{"value": ip} for ip in kv_config.allowed_ips],
                "virtualNetworkRules": [{"id": vnet} for vnet in kv_config.virtual_networks]
            }
        },
        "tags": kv_config.tags
    }
    
    with open("security/templates/keyvault.json", "w") as f:
        json.dump(kv_template, f, indent=2)
    
    # RBAC templates
    custom_role = rbac_manager.generate_custom_role_definition()
    with open("security/templates/custom-role.json", "w") as f:
        json.dump(custom_role, f, indent=2)
    
    # Network security templates
    vnet_template = network_manager.generate_vnet_template(network_config)
    with open("security/templates/vnet.json", "w") as f:
        json.dump(vnet_template, f, indent=2)
    
    nsg_template = network_manager.generate_nsg_template(network_config)
    with open("security/templates/nsg.json", "w") as f:
        json.dump(nsg_template, f, indent=2)
    
    # Compliance frameworks
    print("📋 Generating compliance documentation...")
    compliance_manager = ComplianceManager()
    
    os.makedirs("security/compliance", exist_ok=True)
    
    soc2_controls = compliance_manager.generate_soc2_controls()
    with open("security/compliance/soc2-controls.json", "w") as f:
        json.dump(soc2_controls, f, indent=2)
    
    hipaa_controls = compliance_manager.generate_hipaa_controls()
    with open("security/compliance/hipaa-controls.json", "w") as f:
        json.dump(hipaa_controls, f, indent=2)
    
    compliance_queries = compliance_manager.generate_compliance_monitoring_queries()
    with open("security/compliance/monitoring-queries.json", "w") as f:
        json.dump(compliance_queries, f, indent=2)
    
    # Security monitoring
    print("🛡️ Setting up security monitoring...")
    security_manager = SecurityMonitoringManager()
    
    os.makedirs("security/monitoring", exist_ok=True)
    
    security_alerts = security_manager.generate_security_alerts()
    with open("security/monitoring/security-alerts.json", "w") as f:
        json.dump(security_alerts, f, indent=2)
    
    security_dashboard = security_manager.generate_security_dashboard()
    with open("security/monitoring/security-dashboard.json", "w") as f:
        json.dump(security_dashboard, f, indent=2)
    
    # Data protection policy
    # data_protection = DataProtectionManager(kv_manager)
    data_policy = {
        "version": "1.0",
        "dataClassification": {
            "public": {"encryptionRequired": False},
            "internal": {"encryptionRequired": True},
            "confidential": {"encryptionRequired": True, "additionalControls": ["DLP"]},
            "restricted": {"encryptionRequired": True, "additionalControls": ["DLP", "MFA"]}
        }
    }
    
    with open("security/data-protection-policy.json", "w") as f:
        json.dump(data_policy, f, indent=2)
    
    print("\n✅ Security and compliance setup completed!")
    print("\n📁 Generated Files:")
    print("├── security/")
    print("│   ├── templates/")
    print("│   │   ├── keyvault.json")
    print("│   │   ├── custom-role.json")
    print("│   │   ├── vnet.json")
    print("│   │   └── nsg.json")
    print("│   ├── compliance/")
    print("│   │   ├── soc2-controls.json")
    print("│   │   ├── hipaa-controls.json")
    print("│   │   └── monitoring-queries.json")
    print("│   ├── monitoring/")
    print("│   │   ├── security-alerts.json")
    print("│   │   └── security-dashboard.json")
    print("│   └── data-protection-policy.json")
    
    print("\n🚀 Next Steps:")
    print("1. Deploy Key Vault and configure access policies")
    print("2. Set up Managed Identity and RBAC roles") 
    print("3. Configure network security and private endpoints")
    print("4. Implement data classification and encryption")
    print("5. Deploy security monitoring and alerting")
    print("6. Conduct compliance audits and assessments")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_security_compliance())
