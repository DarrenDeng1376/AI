"""
Azure Kubernetes Service (AKS) Deployment for LangChain Applications

This module provides comprehensive examples for deploying LangChain applications
on Azure Kubernetes Service with enterprise-grade features including auto-scaling,
monitoring, security, and CI/CD integration.

Key components covered:
1. AKS cluster configuration with best practices
2. Kubernetes manifests for LangChain applications
3. Helm charts for complex deployments
4. Azure Container Registry integration
5. Monitoring and logging with Azure Monitor
6. Ingress controllers and SSL termination
7. Horizontal Pod Autoscaling (HPA)
8. Azure DevOps CI/CD pipelines
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
from azure.mgmt.containerregistry import ContainerRegistryManagementClient


# ============================================================================
# 1. AKS CLUSTER CONFIGURATION
# ============================================================================

@dataclass
class AKSClusterConfig:
    """AKS cluster configuration"""
    cluster_name: str
    resource_group: str
    location: str = "East US"
    kubernetes_version: str = "1.28.0"
    node_count: int = 3
    min_node_count: int = 1
    max_node_count: int = 10
    vm_size: str = "Standard_D4s_v3"
    disk_size_gb: int = 128
    network_plugin: str = "azure"
    enable_auto_scaling: bool = True
    enable_azure_policy: bool = True
    enable_monitoring: bool = True
    enable_workload_identity: bool = True
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {
                "Environment": "production",
                "Application": "langchain-ai",
                "ManagedBy": "AKS",
                "CostCenter": "AI-ML"
            }

@dataclass
class ContainerRegistryConfig:
    """Azure Container Registry configuration"""
    registry_name: str
    resource_group: str
    location: str = "East US"
    sku: str = "Premium"
    admin_enabled: bool = False
    public_network_access: bool = False
    tags: Dict[str, str] = None


# ============================================================================
# 2. KUBERNETES MANIFEST GENERATORS
# ============================================================================

class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for LangChain applications"""
    
    def __init__(self, app_name: str, namespace: str = "langchain"):
        self.app_name = app_name
        self.namespace = namespace
    
    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace,
                "labels": {
                    "name": self.namespace,
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/part-of": "langchain-platform"
                }
            }
        }
    
    def generate_configmap(self, config_data: Dict[str, str]) -> Dict[str, Any]:
        """Generate ConfigMap for application configuration"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.app_name}-config",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "config"
                }
            },
            "data": config_data
        }
    
    def generate_secret(self, secret_data: Dict[str, str]) -> Dict[str, Any]:
        """Generate Secret for sensitive configuration"""
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{self.app_name}-secrets",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "secrets"
                }
            },
            "type": "Opaque",
            "data": secret_data  # Base64 encoded values
        }
    
    def generate_deployment(self, 
                          image: str, 
                          replicas: int = 3,
                          resources: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate Deployment manifest"""
        if resources is None:
            resources = {
                "requests": {"cpu": "250m", "memory": "512Mi"},
                "limits": {"cpu": "1000m", "memory": "2Gi"}
            }
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.app_name}-deployment",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "api",
                    "app.kubernetes.io/version": "v1.0.0"
                }
            },
            "spec": {
                "replicas": replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": self.app_name,
                        "app.kubernetes.io/component": "api"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": self.app_name,
                            "app.kubernetes.io/component": "api",
                            "app.kubernetes.io/version": "v1.0.0"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8000",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": f"{self.app_name}-sa",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        },
                        "containers": [
                            {
                                "name": self.app_name,
                                "image": image,
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {
                                        "name": "http",
                                        "containerPort": 8000,
                                        "protocol": "TCP"
                                    }
                                ],
                                "envFrom": [
                                    {
                                        "configMapRef": {
                                            "name": f"{self.app_name}-config"
                                        }
                                    },
                                    {
                                        "secretRef": {
                                            "name": f"{self.app_name}-secrets"
                                        }
                                    }
                                ],
                                "resources": resources,
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": "http"
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {
                                        "drop": ["ALL"]
                                    }
                                },
                                "volumeMounts": [
                                    {
                                        "name": "tmp",
                                        "mountPath": "/tmp"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "tmp",
                                "emptyDir": {}
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_service(self, port: int = 80, target_port: int = 8000) -> Dict[str, Any]:
        """Generate Service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.app_name}-service",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "service"
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "name": "http",
                        "port": port,
                        "targetPort": target_port,
                        "protocol": "TCP"
                    }
                ],
                "selector": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "api"
                }
            }
        }
    
    def generate_ingress(self, 
                        host: str, 
                        tls_secret: str = None,
                        ingress_class: str = "nginx") -> Dict[str, Any]:
        """Generate Ingress manifest"""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.app_name}-ingress",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "ingress"
                },
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "nginx.ingress.kubernetes.io/force-ssl-redirect": "true",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "ingressClassName": ingress_class,
                "rules": [
                    {
                        "host": host,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{self.app_name}-service",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        if tls_secret:
            ingress["spec"]["tls"] = [
                {
                    "hosts": [host],
                    "secretName": tls_secret
                }
            ]
        
        return ingress
    
    def generate_hpa(self, 
                    min_replicas: int = 2, 
                    max_replicas: int = 20,
                    cpu_threshold: int = 70,
                    memory_threshold: int = 80) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.app_name}-hpa",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "autoscaler"
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.app_name}-deployment"
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": cpu_threshold
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": memory_threshold
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 15
                            }
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 10,
                                "periodSeconds": 60
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_service_account(self) -> Dict[str, Any]:
        """Generate ServiceAccount with Azure Workload Identity"""
        return {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": f"{self.app_name}-sa",
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": self.app_name,
                    "app.kubernetes.io/component": "identity"
                },
                "annotations": {
                    "azure.workload.identity/client-id": "your-managed-identity-client-id"
                }
            }
        }
    
    def generate_network_policy(self) -> Dict[str, Any]:
        """Generate NetworkPolicy for security"""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.app_name}-netpol",
                "namespace": self.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": self.app_name
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "ingress-nginx"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 8000
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [],
                        "ports": [
                            {"protocol": "TCP", "port": 443},  # HTTPS
                            {"protocol": "TCP", "port": 53},   # DNS
                            {"protocol": "UDP", "port": 53}    # DNS
                        ]
                    }
                ]
            }
        }


# ============================================================================
# 3. HELM CHART GENERATOR
# ============================================================================

class HelmChartGenerator:
    """Generate Helm charts for complex LangChain deployments"""
    
    def __init__(self, chart_name: str, chart_version: str = "0.1.0"):
        self.chart_name = chart_name
        self.chart_version = chart_version
    
    def generate_chart_yaml(self) -> str:
        """Generate Chart.yaml"""
        return f'''apiVersion: v2
name: {self.chart_name}
description: A Helm chart for LangChain applications on Azure AKS
type: application
version: {self.chart_version}
appVersion: "1.0.0"

keywords:
  - langchain
  - ai
  - azure
  - kubernetes

maintainers:
  - name: AI/ML Team
    email: aiml-team@company.com

dependencies:
  - name: nginx-ingress
    version: "4.7.1"
    repository: "https://kubernetes.github.io/ingress-nginx"
    condition: ingress.enabled
  
  - name: cert-manager
    version: "1.12.0"
    repository: "https://charts.jetstack.io"
    condition: certManager.enabled
  
  - name: prometheus
    version: "23.1.0"
    repository: "https://prometheus-community.github.io/helm-charts"
    condition: monitoring.prometheus.enabled
'''

    def generate_values_yaml(self) -> str:
        """Generate values.yaml with comprehensive configuration"""
        return f'''# Default values for {self.chart_name}
# This is a YAML-formatted file.

replicaCount: 3

image:
  repository: your-acr.azurecr.io/{self.chart_name}
  pullPolicy: Always
  tag: "latest"

imagePullSecrets:
  - name: acr-secret

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations:
    azure.workload.identity/client-id: "your-managed-identity-client-id"
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
  hosts:
    - host: {self.chart_name}.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: {self.chart_name}-tls
      hosts:
        - {self.chart_name}.example.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {{}}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - {self.chart_name}
        topologyKey: kubernetes.io/hostname

# Application configuration
config:
  logLevel: "INFO"
  port: 8000
  workers: 4
  timeout: 300
  
# Azure-specific configuration
azure:
  openai:
    endpoint: ""
    apiVersion: "2023-12-01-preview"
  cosmosdb:
    endpoint: ""
    database: "langchain_db"
    container: "embeddings"
  keyvault:
    name: ""
  
# Monitoring configuration
monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s
      path: /metrics
  
  applicationInsights:
    enabled: true
    connectionString: ""

# Security configuration
security:
  networkPolicy:
    enabled: true
  podDisruptionBudget:
    enabled: true
    minAvailable: 1

# Certificate management
certManager:
  enabled: true
  issuer:
    email: admin@example.com
    server: https://acme-v02.api.letsencrypt.org/directory

# Redis cache (optional)
redis:
  enabled: false
  auth:
    enabled: true
  master:
    persistence:
      enabled: true
      size: 8Gi
'''

    def generate_deployment_template(self) -> str:
        """Generate Helm deployment template"""
        return '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "langchain.fullname" . }}
  labels:
    {{- include "langchain.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      {{- include "langchain.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "langchain.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "langchain.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          env:
            - name: LOG_LEVEL
              value: {{ .Values.config.logLevel | quote }}
            - name: PORT
              value: {{ .Values.config.port | quote }}
            - name: WORKERS
              value: {{ .Values.config.workers | quote }}
            - name: AZURE_OPENAI_ENDPOINT
              valueFrom:
                secretKeyRef:
                  name: {{ include "langchain.fullname" . }}-secrets
                  key: azure-openai-endpoint
            - name: AZURE_OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "langchain.fullname" . }}-secrets
                  key: azure-openai-api-key
            - name: COSMOS_DB_CONNECTION_STRING
              valueFrom:
                secretKeyRef:
                  name: {{ include "langchain.fullname" . }}-secrets
                  key: cosmos-db-connection-string
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
'''


# ============================================================================
# 4. AZURE DEVOPS PIPELINE GENERATOR
# ============================================================================

class AzureDevOpsPipelineGenerator:
    """Generate Azure DevOps CI/CD pipelines for AKS deployment"""
    
    def __init__(self, app_name: str, subscription_id: str, resource_group: str):
        self.app_name = app_name
        self.subscription_id = subscription_id
        self.resource_group = resource_group
    
    def generate_ci_pipeline(self) -> str:
        """Generate CI pipeline YAML"""
        return f'''# Azure DevOps CI Pipeline for {self.app_name}
trigger:
  branches:
    include:
    - main
    - develop
  paths:
    include:
    - src/*
    - Dockerfile
    - requirements.txt

variables:
  - group: {self.app_name}-variables
  - name: imageRepository
    value: '{self.app_name}'
  - name: containerRegistry
    value: 'your-acr.azurecr.io'
  - name: dockerfilePath
    value: '$(Build.SourcesDirectory)/Dockerfile'
  - name: tag
    value: '$(Build.BuildId)'
  - name: vmImageName
    value: 'ubuntu-latest'

stages:
- stage: Test
  displayName: Test and Quality
  jobs:
  - job: UnitTests
    displayName: Run Unit Tests
    pool:
      vmImage: $(vmImageName)
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
        displayName: 'Use Python 3.11'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/ --junitxml=junit/test-results.xml --cov=src --cov-report=xml
      displayName: 'Run tests with coverage'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: '**/test-*.xml'
        testRunTitle: 'Publish test results for Python $(python.version)'
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: '$(System.DefaultWorkingDirectory)/**/coverage.xml'

  - job: SecurityScan
    displayName: Security Scanning
    pool:
      vmImage: $(vmImageName)
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
    
    - script: |
        pip install safety bandit
        safety check -r requirements.txt
        bandit -r src/ -f json -o bandit-report.json
      displayName: 'Security scanning'
      continueOnError: true
    
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: 'bandit-report.json'
        artifactName: 'security-report'

- stage: Build
  displayName: Build and Push
  dependsOn: Test
  condition: succeeded()
  jobs:
  - job: Build
    displayName: Build and Push Docker Image
    pool:
      vmImage: $(vmImageName)
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: 'ACR-ServiceConnection'
        tags: |
          $(tag)
          latest
    
    - task: AzureCLI@2
      displayName: 'Scan image for vulnerabilities'
      inputs:
        azureSubscription: 'Azure-ServiceConnection'
        scriptType: 'bash'
        scriptLocation: 'inlineScript'
        inlineScript: |
          az acr run --registry $(containerRegistry) --cmd "trivy image $(containerRegistry)/$(imageRepository):$(tag)" /dev/null

- stage: DeployDev
  displayName: Deploy to Development
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
  variables:
  - group: {self.app_name}-dev-variables
  jobs:
  - deployment: DeployToDev
    displayName: Deploy to AKS Development
    pool:
      vmImage: $(vmImageName)
    environment: 'development'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: Deploy to Kubernetes cluster
            inputs:
              action: deploy
              kubernetesServiceConnection: 'AKS-Dev-ServiceConnection'
              namespace: '{self.app_name}-dev'
              manifests: |
                $(Pipeline.Workspace)/manifests/deployment.yaml
                $(Pipeline.Workspace)/manifests/service.yaml
              containers: |
                $(containerRegistry)/$(imageRepository):$(tag)

- stage: DeployProd
  displayName: Deploy to Production
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  variables:
  - group: {self.app_name}-prod-variables
  jobs:
  - deployment: DeployToProd
    displayName: Deploy to AKS Production
    pool:
      vmImage: $(vmImageName)
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: HelmDeploy@0
            displayName: Helm upgrade
            inputs:
              connectionType: 'Kubernetes Service Connection'
              kubernetesServiceConnection: 'AKS-Prod-ServiceConnection'
              namespace: '{self.app_name}'
              command: 'upgrade'
              chartType: 'FilePath'
              chartPath: '$(Pipeline.Workspace)/helm/{self.app_name}'
              releaseName: '{self.app_name}'
              valueFile: '$(Pipeline.Workspace)/helm/{self.app_name}/values-prod.yaml'
              overrideValues: |
                image.tag=$(tag)
                ingress.hosts[0].host={self.app_name}.example.com
'''

    def generate_cd_pipeline(self) -> str:
        """Generate CD pipeline YAML for GitOps"""
        return f'''# Azure DevOps CD Pipeline for {self.app_name}
trigger: none

resources:
  pipelines:
  - pipeline: ci-pipeline
    source: '{self.app_name}-ci'
    trigger:
      branches:
        include:
        - main

variables:
- group: {self.app_name}-gitops

stages:
- stage: UpdateManifests
  displayName: Update GitOps Manifests
  jobs:
  - job: UpdateGitOps
    displayName: Update GitOps Repository
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - checkout: self
      persistCredentials: true
      clean: true
    
    - task: AzureCLI@2
      displayName: 'Get new image tag'
      inputs:
        azureSubscription: 'Azure-ServiceConnection'
        scriptType: 'bash'
        scriptLocation: 'inlineScript'
        inlineScript: |
          # Get the latest build ID from the triggering pipeline
          NEW_TAG=$(resources.pipeline.ci-pipeline.runID)
          echo "##vso[task.setvariable variable=newImageTag]$NEW_TAG"
    
    - task: YamlPatch@3
      displayName: 'Update image tag in values.yaml'
      inputs:
        files: 'helm/{self.app_name}/values.yaml'
        yamlOperations: |
          - operation: 'replace'
            path: '$.image.tag'
            value: '$(newImageTag)'
    
    - script: |
        git config user.name "Azure DevOps"
        git config user.email "devops@company.com"
        git add .
        git commit -m "Update image tag to $(newImageTag) [skip ci]"
        git push origin HEAD:main
      displayName: 'Commit and push changes'

- stage: DeployToProduction
  displayName: Deploy to Production
  dependsOn: UpdateManifests
  jobs:
  - deployment: ProductionDeployment
    displayName: Production Deployment
    pool:
      vmImage: 'ubuntu-latest'
    environment: 'production-aks'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: HelmDeploy@0
            displayName: 'Deploy with Helm'
            inputs:
              connectionType: 'Kubernetes Service Connection'
              kubernetesServiceConnection: 'AKS-Prod-ServiceConnection'
              namespace: '{self.app_name}'
              command: 'upgrade'
              chartType: 'FilePath'
              chartPath: 'helm/{self.app_name}'
              releaseName: '{self.app_name}'
              install: true
              waitForExecution: true
              arguments: '--timeout 10m'
          
          - task: Kubernetes@1
            displayName: 'Check deployment status'
            inputs:
              connectionType: 'Kubernetes Service Connection'
              kubernetesServiceConnection: 'AKS-Prod-ServiceConnection'
              namespace: '{self.app_name}'
              command: 'get'
              arguments: 'pods -l app.kubernetes.io/name={self.app_name}'
'''


# ============================================================================
# 5. MONITORING AND OBSERVABILITY
# ============================================================================

class MonitoringManifestGenerator:
    """Generate monitoring and observability manifests"""
    
    def generate_prometheus_servicemonitor(self, app_name: str, namespace: str) -> Dict[str, Any]:
        """Generate Prometheus ServiceMonitor"""
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{app_name}-metrics",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": app_name,
                    "app.kubernetes.io/component": "monitoring"
                }
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": app_name
                    }
                },
                "endpoints": [
                    {
                        "port": "http",
                        "path": "/metrics",
                        "interval": "30s",
                        "scrapeTimeout": "10s"
                    }
                ]
            }
        }
    
    def generate_grafana_dashboard_configmap(self, app_name: str, namespace: str) -> Dict[str, Any]:
        """Generate Grafana dashboard ConfigMap"""
        dashboard_json = {
            "dashboard": {
                "id": None,
                "title": f"{app_name} Dashboard",
                "tags": ["langchain", "ai", "azure"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f"rate(http_requests_total{{job=\"{app_name}\"}}[5m])",
                                "legendFormat": "{{method}} {{status}}"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f"histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job=\"{app_name}\"}}[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Pod CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f"rate(container_cpu_usage_seconds_total{{pod=~\"{app_name}-.*\"}}[5m])",
                                "legendFormat": "{{pod}}"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Pod Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f"container_memory_usage_bytes{{pod=~\"{app_name}-.*\"}}",
                                "legendFormat": "{{pod}}"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{app_name}-dashboard",
                "namespace": namespace,
                "labels": {
                    "grafana_dashboard": "1",
                    "app.kubernetes.io/name": app_name
                }
            },
            "data": {
                f"{app_name}-dashboard.json": json.dumps(dashboard_json, indent=2)
            }
        }


# ============================================================================
# 6. DEMONSTRATION AND USAGE
# ============================================================================

def demonstrate_aks_deployment():
    """Demonstrate AKS deployment generation"""
    print("🚀 Azure Kubernetes Service (AKS) Deployment Demo")
    print("=" * 60)
    
    app_name = "langchain-ai-app"
    namespace = "langchain"
    
    # Generate Kubernetes manifests
    print("📋 Generating Kubernetes Manifests...")
    k8s_gen = KubernetesManifestGenerator(app_name, namespace)
    
    os.makedirs("k8s/base", exist_ok=True)
    
    manifests = {
        "namespace.yaml": k8s_gen.generate_namespace(),
        "configmap.yaml": k8s_gen.generate_configmap({
            "LOG_LEVEL": "INFO",
            "PORT": "8000",
            "WORKERS": "4"
        }),
        "secret.yaml": k8s_gen.generate_secret({
            "azure-openai-key": "your-base64-encoded-key",
            "cosmos-connection": "your-base64-encoded-connection-string"
        }),
        "deployment.yaml": k8s_gen.generate_deployment(
            "your-acr.azurecr.io/langchain-ai-app:latest"
        ),
        "service.yaml": k8s_gen.generate_service(),
        "ingress.yaml": k8s_gen.generate_ingress("langchain-ai.example.com"),
        "hpa.yaml": k8s_gen.generate_hpa(),
        "serviceaccount.yaml": k8s_gen.generate_service_account(),
        "networkpolicy.yaml": k8s_gen.generate_network_policy()
    }
    
    for filename, manifest in manifests.items():
        with open(f"k8s/base/{filename}", "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Kubernetes manifests saved to k8s/base/")
    
    # Generate Helm chart
    print("📋 Generating Helm Chart...")
    helm_gen = HelmChartGenerator(app_name)
    
    os.makedirs(f"helm/{app_name}/templates", exist_ok=True)
    
    with open(f"helm/{app_name}/Chart.yaml", "w") as f:
        f.write(helm_gen.generate_chart_yaml())
    
    with open(f"helm/{app_name}/values.yaml", "w") as f:
        f.write(helm_gen.generate_values_yaml())
    
    with open(f"helm/{app_name}/templates/deployment.yaml", "w") as f:
        f.write(helm_gen.generate_deployment_template())
    
    print("✅ Helm chart saved to helm/ directory")
    
    # Generate Azure DevOps pipelines
    print("📋 Generating Azure DevOps Pipelines...")
    devops_gen = AzureDevOpsPipelineGenerator(
        app_name, 
        "your-subscription-id", 
        "your-resource-group"
    )
    
    os.makedirs("pipelines", exist_ok=True)
    
    with open("pipelines/ci-pipeline.yml", "w") as f:
        f.write(devops_gen.generate_ci_pipeline())
    
    with open("pipelines/cd-pipeline.yml", "w") as f:
        f.write(devops_gen.generate_cd_pipeline())
    
    print("✅ Azure DevOps pipelines saved to pipelines/ directory")
    
    # Generate monitoring manifests
    print("📋 Generating Monitoring Manifests...")
    monitoring_gen = MonitoringManifestGenerator()
    
    os.makedirs("k8s/monitoring", exist_ok=True)
    
    servicemonitor = monitoring_gen.generate_prometheus_servicemonitor(app_name, namespace)
    dashboard_cm = monitoring_gen.generate_grafana_dashboard_configmap(app_name, namespace)
    
    with open("k8s/monitoring/servicemonitor.yaml", "w") as f:
        yaml.dump(servicemonitor, f, default_flow_style=False)
    
    with open("k8s/monitoring/grafana-dashboard.yaml", "w") as f:
        yaml.dump(dashboard_cm, f, default_flow_style=False)
    
    print("✅ Monitoring manifests saved to k8s/monitoring/")
    
    print("\n🎉 AKS deployment generation completed!")
    print("\n📁 Generated Files:")
    print("├── k8s/")
    print("│   ├── base/")
    print("│   │   ├── namespace.yaml")
    print("│   │   ├── configmap.yaml")
    print("│   │   ├── secret.yaml")
    print("│   │   ├── deployment.yaml")
    print("│   │   ├── service.yaml")
    print("│   │   ├── ingress.yaml")
    print("│   │   ├── hpa.yaml")
    print("│   │   ├── serviceaccount.yaml")
    print("│   │   └── networkpolicy.yaml")
    print("│   └── monitoring/")
    print("│       ├── servicemonitor.yaml")
    print("│       └── grafana-dashboard.yaml")
    print("├── helm/")
    print(f"│   └── {app_name}/")
    print("│       ├── Chart.yaml")
    print("│       ├── values.yaml")
    print("│       └── templates/")
    print("│           └── deployment.yaml")
    print("└── pipelines/")
    print("    ├── ci-pipeline.yml")
    print("    └── cd-pipeline.yml")
    
    print("\n🚀 Next Steps:")
    print("1. Customize the generated manifests for your environment")
    print("2. Update image repositories and secrets")
    print("3. Deploy to AKS using kubectl or Helm")
    print("4. Set up Azure DevOps service connections")
    print("5. Configure monitoring and alerting")


if __name__ == "__main__":
    demonstrate_aks_deployment()
