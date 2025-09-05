"""
Azure Cost Optimization and Auto-Scaling for LangChain Applications

This module provides comprehensive cost optimization and auto-scaling solutions
for LangChain applications on Azure. It includes cost monitoring, resource
optimization, auto-scaling configurations, and cost management policies.

Key features:
1. Azure Cost Management and billing analysis
2. Auto-scaling policies for compute resources
3. Resource right-sizing recommendations
4. Cost optimization strategies
5. Budget alerts and cost controls
6. Performance vs cost optimization
7. Reserved instance recommendations
8. Spot instance management
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from azure.identity import DefaultAzureCredential
import asyncio
import time


# ============================================================================
# 1. COST MONITORING CONFIGURATION
# ============================================================================

@dataclass
class CostMonitoringConfig:
    """Cost monitoring configuration"""
    subscription_id: str
    resource_group: str
    cost_center: str
    budget_amount: float
    budget_period: str = "monthly"  # monthly, quarterly, annually
    currency: str = "USD"
    alert_thresholds: List[int] = None  # Percentage thresholds
    notification_emails: List[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = [50, 80, 90, 100]
        if self.notification_emails is None:
            self.notification_emails = ["admin@company.com"]
        if self.tags is None:
            self.tags = {
                "CostCenter": self.cost_center,
                "Environment": "production",
                "Application": "langchain-ai"
            }


class CostAnalysisManager:
    """Manage cost analysis and monitoring"""
    
    def __init__(self, config: CostMonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_budget_alert(self) -> Dict[str, Any]:
        """Generate budget alert ARM template"""
        return {
            "type": "Microsoft.Consumption/budgets",
            "apiVersion": "2021-10-01",
            "name": f"langchain-{self.config.budget_period}-budget",
            "scope": f"/subscriptions/{self.config.subscription_id}/resourceGroups/{self.config.resource_group}",
            "properties": {
                "timePeriod": {
                    "startDate": datetime.now().strftime("%Y-%m-01"),
                    "endDate": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-01")
                },
                "timeGrain": self.config.budget_period.capitalize(),
                "amount": self.config.budget_amount,
                "category": "Cost",
                "filters": {
                    "resourceGroups": [self.config.resource_group],
                    "tags": self.config.tags
                },
                "notifications": {
                    f"notification{threshold}": {
                        "enabled": True,
                        "operator": "GreaterThan",
                        "threshold": threshold,
                        "contactEmails": self.config.notification_emails,
                        "contactRoles": ["Owner", "Contributor"],
                        "thresholdType": "Actual" if threshold <= 100 else "Forecasted"
                    } for threshold in self.config.alert_thresholds
                }
            }
        }
    
    def generate_cost_analysis_queries(self) -> Dict[str, str]:
        """Generate KQL queries for cost analysis"""
        return {
            "daily_costs": f"""
                Usage
                | where TimeGenerated >= ago(30d)
                | where ResourceGroup == "{self.config.resource_group}"
                | summarize DailyCost = sum(Cost) by bin(TimeGenerated, 1d), ResourceType
                | render timechart
            """,
            "cost_by_service": f"""
                Usage
                | where TimeGenerated >= ago(7d)
                | where ResourceGroup == "{self.config.resource_group}"
                | summarize TotalCost = sum(Cost) by MeterCategory
                | order by TotalCost desc
                | render piechart
            """,
            "cost_trends": f"""
                Usage
                | where TimeGenerated >= ago(90d)
                | where ResourceGroup == "{self.config.resource_group}"
                | summarize WeeklyCost = sum(Cost) by bin(TimeGenerated, 7d)
                | extend CostTrend = case(
                    WeeklyCost > prev(WeeklyCost), "Increasing",
                    WeeklyCost < prev(WeeklyCost), "Decreasing", 
                    "Stable"
                )
                | render timechart
            """,
            "top_cost_resources": f"""
                Usage
                | where TimeGenerated >= ago(7d)
                | where ResourceGroup == "{self.config.resource_group}"
                | summarize ResourceCost = sum(Cost) by ResourceName, ResourceType
                | order by ResourceCost desc
                | take 10
            """
        }
    
    def generate_cost_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate cost optimization recommendations"""
        return {
            "compute_optimization": {
                "description": "Right-size compute resources based on utilization",
                "strategies": [
                    {
                        "name": "VM Right-sizing",
                        "description": "Analyze CPU and memory utilization to recommend appropriate VM sizes",
                        "potential_savings": "15-30%",
                        "implementation": "Use Azure Advisor recommendations"
                    },
                    {
                        "name": "Auto-shutdown",
                        "description": "Automatically shutdown non-production VMs during off-hours",
                        "potential_savings": "50-70%",
                        "implementation": "Azure DevTest Labs auto-shutdown policies"
                    },
                    {
                        "name": "Spot Instances",
                        "description": "Use Azure Spot VMs for non-critical workloads",
                        "potential_savings": "60-90%",
                        "implementation": "Azure Spot VM scale sets"
                    }
                ]
            },
            "storage_optimization": {
                "description": "Optimize storage costs through tiering and lifecycle management",
                "strategies": [
                    {
                        "name": "Storage Tiering",
                        "description": "Move infrequently accessed data to cooler storage tiers",
                        "potential_savings": "40-80%",
                        "implementation": "Azure Storage lifecycle management"
                    },
                    {
                        "name": "Data Compression",
                        "description": "Compress data before storage to reduce storage costs",
                        "potential_savings": "20-50%",
                        "implementation": "Application-level compression"
                    }
                ]
            },
            "ai_service_optimization": {
                "description": "Optimize AI service usage and costs",
                "strategies": [
                    {
                        "name": "Model Selection",
                        "description": "Use appropriate model sizes for different use cases",
                        "potential_savings": "30-60%",
                        "implementation": "GPT-3.5-turbo for simpler tasks, GPT-4 for complex reasoning"
                    },
                    {
                        "name": "Caching",
                        "description": "Cache frequently requested results to reduce API calls",
                        "potential_savings": "40-70%",
                        "implementation": "Redis cache for responses and embeddings"
                    },
                    {
                        "name": "Batch Processing",
                        "description": "Batch multiple requests to reduce per-request overhead",
                        "potential_savings": "20-40%",
                        "implementation": "Queue requests and process in batches"
                    }
                ]
            }
        }


# ============================================================================
# 2. AUTO-SCALING CONFIGURATION
# ============================================================================

@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""
    resource_type: str  # "vm_scale_set", "app_service", "aks", "function_app"
    min_instances: int = 1
    max_instances: int = 10
    default_instances: int = 2
    scale_out_threshold: int = 70  # CPU percentage
    scale_in_threshold: int = 30   # CPU percentage
    scale_out_duration: int = 5    # minutes
    scale_in_duration: int = 10    # minutes
    scale_out_cooldown: int = 5    # minutes
    scale_in_cooldown: int = 10    # minutes
    custom_metrics: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = [
                {
                    "name": "langchain_active_requests",
                    "threshold": 100,
                    "operator": "GreaterThan",
                    "action": "scale_out"
                },
                {
                    "name": "langchain_queue_depth",
                    "threshold": 50,
                    "operator": "GreaterThan", 
                    "action": "scale_out"
                }
            ]


class AutoScalingManager:
    """Manage auto-scaling configurations"""
    
    def generate_vmss_autoscale_settings(self, config: AutoScalingConfig) -> Dict[str, Any]:
        """Generate VM Scale Set auto-scale settings"""
        return {
            "type": "Microsoft.Insights/autoscalesettings",
            "apiVersion": "2022-10-01",
            "name": "langchain-vmss-autoscale",
            "location": "[resourceGroup().location]",
            "properties": {
                "profiles": [
                    {
                        "name": "defaultProfile",
                        "capacity": {
                            "minimum": str(config.min_instances),
                            "maximum": str(config.max_instances),
                            "default": str(config.default_instances)
                        },
                        "rules": [
                            {
                                "metricTrigger": {
                                    "metricName": "Percentage CPU",
                                    "metricNamespace": "Microsoft.Compute/virtualMachineScaleSets",
                                    "metricResourceUri": "[resourceId('Microsoft.Compute/virtualMachineScaleSets', 'langchain-vmss')]",
                                    "timeGrain": "PT1M",
                                    "statistic": "Average",
                                    "timeWindow": f"PT{config.scale_out_duration}M",
                                    "timeAggregation": "Average",
                                    "operator": "GreaterThan",
                                    "threshold": config.scale_out_threshold
                                },
                                "scaleAction": {
                                    "direction": "Increase",
                                    "type": "ChangeCount",
                                    "value": "1",
                                    "cooldown": f"PT{config.scale_out_cooldown}M"
                                }
                            },
                            {
                                "metricTrigger": {
                                    "metricName": "Percentage CPU",
                                    "metricNamespace": "Microsoft.Compute/virtualMachineScaleSets",
                                    "metricResourceUri": "[resourceId('Microsoft.Compute/virtualMachineScaleSets', 'langchain-vmss')]",
                                    "timeGrain": "PT1M",
                                    "statistic": "Average",
                                    "timeWindow": f"PT{config.scale_in_duration}M",
                                    "timeAggregation": "Average",
                                    "operator": "LessThan",
                                    "threshold": config.scale_in_threshold
                                },
                                "scaleAction": {
                                    "direction": "Decrease",
                                    "type": "ChangeCount",
                                    "value": "1",
                                    "cooldown": f"PT{config.scale_in_cooldown}M"
                                }
                            }
                        ]
                    }
                ],
                "enabled": True,
                "targetResourceUri": "[resourceId('Microsoft.Compute/virtualMachineScaleSets', 'langchain-vmss')]"
            }
        }
    
    def generate_app_service_autoscale_settings(self, config: AutoScalingConfig) -> Dict[str, Any]:
        """Generate App Service auto-scale settings"""
        return {
            "type": "Microsoft.Insights/autoscalesettings",
            "apiVersion": "2022-10-01",
            "name": "langchain-app-autoscale",
            "location": "[resourceGroup().location]",
            "properties": {
                "profiles": [
                    {
                        "name": "defaultProfile",
                        "capacity": {
                            "minimum": str(config.min_instances),
                            "maximum": str(config.max_instances),
                            "default": str(config.default_instances)
                        },
                        "rules": [
                            {
                                "metricTrigger": {
                                    "metricName": "CpuPercentage",
                                    "metricNamespace": "Microsoft.Web/serverfarms",
                                    "metricResourceUri": "[resourceId('Microsoft.Web/serverfarms', 'langchain-app-plan')]",
                                    "timeGrain": "PT1M",
                                    "statistic": "Average",
                                    "timeWindow": f"PT{config.scale_out_duration}M",
                                    "timeAggregation": "Average",
                                    "operator": "GreaterThan",
                                    "threshold": config.scale_out_threshold
                                },
                                "scaleAction": {
                                    "direction": "Increase",
                                    "type": "ChangeCount",
                                    "value": "1",
                                    "cooldown": f"PT{config.scale_out_cooldown}M"
                                }
                            },
                            {
                                "metricTrigger": {
                                    "metricName": "MemoryPercentage",
                                    "metricNamespace": "Microsoft.Web/serverfarms",
                                    "metricResourceUri": "[resourceId('Microsoft.Web/serverfarms', 'langchain-app-plan')]",
                                    "timeGrain": "PT1M",
                                    "statistic": "Average",
                                    "timeWindow": f"PT{config.scale_out_duration}M",
                                    "timeAggregation": "Average",
                                    "operator": "GreaterThan",
                                    "threshold": 80
                                },
                                "scaleAction": {
                                    "direction": "Increase",
                                    "type": "ChangeCount",
                                    "value": "1",
                                    "cooldown": f"PT{config.scale_out_cooldown}M"
                                }
                            }
                        ]
                    }
                ],
                "enabled": True,
                "targetResourceUri": "[resourceId('Microsoft.Web/serverfarms', 'langchain-app-plan')]"
            }
        }
    
    def generate_aks_hpa_config(self, config: AutoScalingConfig) -> Dict[str, Any]:
        """Generate AKS Horizontal Pod Autoscaler configuration"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "langchain-hpa",
                "namespace": "langchain"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "langchain-deployment"
                },
                "minReplicas": config.min_instances,
                "maxReplicas": config.max_instances,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.scale_out_threshold
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ] + [
                    {
                        "type": "Object",
                        "object": {
                            "metric": {
                                "name": metric["name"]
                            },
                            "target": {
                                "type": "Value",
                                "value": str(metric["threshold"])
                            },
                            "describedObject": {
                                "apiVersion": "v1",
                                "kind": "Service",
                                "name": "langchain-service"
                            }
                        }
                    } for metric in config.custom_metrics
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": config.scale_out_cooldown * 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 15
                            },
                            {
                                "type": "Pods",
                                "value": 2,
                                "periodSeconds": 60
                            }
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": config.scale_in_cooldown * 60,
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


# ============================================================================
# 3. RESOURCE OPTIMIZATION
# ============================================================================

class ResourceOptimizationManager:
    """Manage resource optimization strategies"""
    
    def generate_advisor_recommendations_query(self) -> str:
        """Generate query for Azure Advisor recommendations"""
        return """
            AdvisorResources
            | where TimeGenerated >= ago(7d)
            | where Category in ("Cost", "Performance", "HighAvailability")
            | summarize RecommendationCount = count() by Category, Impact
            | order by Impact desc, RecommendationCount desc
        """
    
    def generate_resource_utilization_analysis(self) -> Dict[str, str]:
        """Generate resource utilization analysis queries"""
        return {
            "vm_cpu_utilization": """
                Perf
                | where TimeGenerated >= ago(7d)
                | where ObjectName == "Processor" and CounterName == "% Processor Time"
                | summarize AvgCPU = avg(CounterValue) by Computer, bin(TimeGenerated, 1h)
                | where AvgCPU < 20 or AvgCPU > 80
            """,
            "vm_memory_utilization": """
                Perf
                | where TimeGenerated >= ago(7d)
                | where ObjectName == "Memory" and CounterName == "Available MBytes"
                | extend MemoryUtilization = 100 - (CounterValue / 1024)
                | summarize AvgMemory = avg(MemoryUtilization) by Computer, bin(TimeGenerated, 1h)
                | where AvgMemory < 30 or AvgMemory > 85
            """,
            "storage_utilization": """
                Perf
                | where TimeGenerated >= ago(7d)
                | where ObjectName == "LogicalDisk" and CounterName == "% Free Space"
                | summarize AvgFreeSpace = avg(CounterValue) by Computer, InstanceName, bin(TimeGenerated, 1h)
                | where AvgFreeSpace < 20 or AvgFreeSpace > 80
            """,
            "app_service_performance": """
                AppServiceHTTPLogs
                | where TimeGenerated >= ago(7d)
                | summarize 
                    AvgResponseTime = avg(TimeTaken),
                    RequestCount = count(),
                    ErrorRate = countif(ScStatus >= 400) * 100.0 / count()
                by bin(TimeGenerated, 1h)
                | where AvgResponseTime > 5000 or ErrorRate > 5
            """
        }
    
    def generate_cost_optimization_policies(self) -> Dict[str, Any]:
        """Generate cost optimization policies"""
        return {
            "vm_shutdown_policy": {
                "name": "VM Auto-shutdown Policy",
                "description": "Automatically shutdown VMs during off-hours",
                "schedule": {
                    "shutdown_time": "18:00",
                    "timezone": "UTC",
                    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "send_notification": True,
                    "notification_time": 30  # minutes before shutdown
                },
                "exceptions": {
                    "production_vms": True,
                    "tagged_vms": ["Environment=Production", "AlwaysOn=True"]
                }
            },
            "storage_lifecycle_policy": {
                "name": "Storage Lifecycle Management",
                "description": "Automatically move data to appropriate storage tiers",
                "rules": [
                    {
                        "name": "move_to_cool",
                        "condition": "last_modified > 30 days",
                        "action": "move to cool tier"
                    },
                    {
                        "name": "move_to_archive",
                        "condition": "last_modified > 365 days",
                        "action": "move to archive tier"
                    },
                    {
                        "name": "delete_old_logs",
                        "condition": "last_modified > 2555 days AND type = logs",
                        "action": "delete"
                    }
                ]
            },
            "reserved_instance_recommendations": {
                "name": "Reserved Instance Analysis",
                "description": "Analyze usage patterns to recommend reserved instances",
                "analysis_period": "90 days",
                "minimum_utilization": 70,
                "recommendation_types": ["1-year", "3-year"],
                "vm_series": ["Standard_D", "Standard_E", "Standard_F"]
            }
        }


# ============================================================================
# 4. PERFORMANCE VS COST OPTIMIZATION
# ============================================================================

class PerformanceCostOptimizer:
    """Optimize performance vs cost trade-offs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance_cost_metrics(self) -> Dict[str, Any]:
        """Analyze performance vs cost metrics"""
        return {
            "ai_model_optimization": {
                "analysis": "Compare cost and performance of different AI models",
                "metrics": [
                    {
                        "model": "gpt-3.5-turbo",
                        "cost_per_1k_tokens": 0.002,
                        "avg_response_time": "2.1s",
                        "quality_score": 8.5,
                        "use_cases": ["Simple Q&A", "Text summarization", "Basic conversations"]
                    },
                    {
                        "model": "gpt-4",
                        "cost_per_1k_tokens": 0.03,
                        "avg_response_time": "4.2s",
                        "quality_score": 9.5,
                        "use_cases": ["Complex reasoning", "Code generation", "Advanced analysis"]
                    },
                    {
                        "model": "gpt-4-turbo",
                        "cost_per_1k_tokens": 0.01,
                        "avg_response_time": "3.1s",
                        "quality_score": 9.3,
                        "use_cases": ["Balanced performance and cost", "Large context windows"]
                    }
                ],
                "recommendations": [
                    "Use GPT-3.5-turbo for 80% of simple queries",
                    "Route complex queries to GPT-4 based on intent classification",
                    "Implement caching to reduce redundant API calls"
                ]
            },
            "compute_optimization": {
                "analysis": "Balance compute performance with cost efficiency",
                "strategies": [
                    {
                        "name": "Burst-capable instances",
                        "description": "Use burstable VMs for variable workloads",
                        "cost_savings": "30-50%",
                        "performance_impact": "Minimal for typical workloads"
                    },
                    {
                        "name": "Spot instances",
                        "description": "Use Azure Spot VMs for batch processing",
                        "cost_savings": "60-90%",
                        "performance_impact": "May have interruptions"
                    },
                    {
                        "name": "Container optimization",
                        "description": "Optimize container resource requests and limits",
                        "cost_savings": "20-40%",
                        "performance_impact": "Better resource utilization"
                    }
                ]
            }
        }
    
    def generate_optimization_dashboard(self) -> Dict[str, Any]:
        """Generate cost optimization dashboard"""
        return {
            "name": "LangChain Cost Optimization Dashboard",
            "panels": [
                {
                    "title": "Cost vs Performance Metrics",
                    "type": "scatter_plot",
                    "x_axis": "Response Time (ms)",
                    "y_axis": "Cost per Request ($)",
                    "size": "Request Volume"
                },
                {
                    "title": "Model Usage Distribution",
                    "type": "pie_chart",
                    "query": """
                        AppTraces
                        | where TimeGenerated >= ago(24h)
                        | extend Model = tostring(customDimensions.model)
                        | summarize RequestCount = count(), TotalCost = sum(toreal(customDimensions.cost)) by Model
                    """
                },
                {
                    "title": "Hourly Cost Trends",
                    "type": "line_chart",
                    "query": """
                        AppTraces
                        | where TimeGenerated >= ago(7d)
                        | extend Cost = toreal(customDimensions.cost)
                        | summarize HourlyCost = sum(Cost) by bin(TimeGenerated, 1h)
                        | render timechart
                    """
                },
                {
                    "title": "Resource Utilization vs Cost",
                    "type": "correlation_chart",
                    "metrics": ["CPU Utilization", "Memory Usage", "Network I/O", "Storage I/O"],
                    "cost_metric": "Hourly Compute Cost"
                }
            ]
        }


# ============================================================================
# 5. SPOT INSTANCE MANAGEMENT
# ============================================================================

class SpotInstanceManager:
    """Manage Azure Spot VM instances for cost optimization"""
    
    def generate_spot_vm_scale_set(self) -> Dict[str, Any]:
        """Generate Spot VM Scale Set configuration"""
        return {
            "type": "Microsoft.Compute/virtualMachineScaleSets",
            "apiVersion": "2023-03-01",
            "name": "langchain-spot-vmss",
            "location": "[resourceGroup().location]",
            "sku": {
                "name": "Standard_D2s_v3",
                "capacity": 2
            },
            "properties": {
                "overprovision": False,
                "upgradePolicy": {
                    "mode": "Manual"
                },
                "virtualMachineProfile": {
                    "priority": "Spot",
                    "evictionPolicy": "Deallocate",
                    "billingProfile": {
                        "maxPrice": 0.05  # Maximum price per hour
                    },
                    "storageProfile": {
                        "osDisk": {
                            "createOption": "FromImage",
                            "caching": "ReadWrite",
                            "managedDisk": {
                                "storageAccountType": "Premium_LRS"
                            }
                        },
                        "imageReference": {
                            "publisher": "Canonical",
                            "offer": "0001-com-ubuntu-server-focal",
                            "sku": "20_04-lts-gen2",
                            "version": "latest"
                        }
                    },
                    "osProfile": {
                        "computerNamePrefix": "langchain",
                        "adminUsername": "azureuser",
                        "adminPassword": "[parameters('adminPassword')]",
                        "customData": "[base64(parameters('cloudInitScript'))]"
                    },
                    "networkProfile": {
                        "networkInterfaceConfigurations": [
                            {
                                "name": "langchain-nic",
                                "properties": {
                                    "primary": True,
                                    "ipConfigurations": [
                                        {
                                            "name": "ipconfig1",
                                            "properties": {
                                                "subnet": {
                                                    "id": "[variables('subnetRef')]"
                                                },
                                                "loadBalancerBackendAddressPools": [
                                                    {
                                                        "id": "[variables('lbBackendPoolRef')]"
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_spot_instance_monitoring(self) -> Dict[str, str]:
        """Generate monitoring queries for Spot instances"""
        return {
            "spot_eviction_rate": """
                AzureActivity
                | where TimeGenerated >= ago(24h)
                | where OperationName == "Microsoft.Compute/virtualMachineScaleSets/virtualmachines/deallocate/action"
                | where Properties contains "Spot"
                | summarize EvictionCount = count() by bin(TimeGenerated, 1h)
                | render timechart
            """,
            "spot_cost_savings": """
                Usage
                | where TimeGenerated >= ago(7d)
                | where MeterCategory == "Virtual Machines"
                | where MeterSubCategory contains "Spot"
                | summarize SpotCost = sum(Cost) by bin(TimeGenerated, 1d)
                | join (
                    Usage
                    | where TimeGenerated >= ago(7d)
                    | where MeterCategory == "Virtual Machines"
                    | where MeterSubCategory !contains "Spot"
                    | summarize RegularCost = sum(Cost) by bin(TimeGenerated, 1d)
                ) on TimeGenerated
                | extend SavingsPercentage = (RegularCost - SpotCost) / RegularCost * 100
                | render timechart
            """,
            "spot_availability": """
                Heartbeat
                | where TimeGenerated >= ago(24h)
                | where Computer contains "langchain"
                | summarize AvailableInstances = dcount(Computer) by bin(TimeGenerated, 15m)
                | render timechart
            """
        }


# ============================================================================
# 6. DEMONSTRATION AND USAGE
# ============================================================================

async def demonstrate_cost_optimization():
    """Demonstrate cost optimization and auto-scaling setup"""
    print("💰 Azure Cost Optimization and Auto-Scaling Demo")
    print("=" * 60)
    
    # Cost monitoring configuration
    cost_config = CostMonitoringConfig(
        subscription_id="your-subscription-id",
        resource_group="langchain-rg",
        cost_center="AI-ML-001",
        budget_amount=5000.0,
        budget_period="monthly"
    )
    
    print("📊 Setting up cost monitoring...")
    cost_manager = CostAnalysisManager(cost_config)
    
    # Auto-scaling configuration
    scaling_config = AutoScalingConfig(
        resource_type="aks",
        min_instances=2,
        max_instances=20,
        scale_out_threshold=70,
        scale_in_threshold=30
    )
    
    print("📈 Configuring auto-scaling...")
    scaling_manager = AutoScalingManager()
    
    # Generate configurations
    print("📋 Generating cost optimization configurations...")
    os.makedirs("cost-optimization/budgets", exist_ok=True)
    os.makedirs("cost-optimization/autoscaling", exist_ok=True)
    os.makedirs("cost-optimization/monitoring", exist_ok=True)
    os.makedirs("cost-optimization/policies", exist_ok=True)
    
    # Budget alerts
    budget_alert = cost_manager.generate_budget_alert()
    with open("cost-optimization/budgets/budget-alert.json", "w") as f:
        json.dump(budget_alert, f, indent=2)
    
    # Cost analysis queries
    cost_queries = cost_manager.generate_cost_analysis_queries()
    with open("cost-optimization/monitoring/cost-queries.json", "w") as f:
        json.dump(cost_queries, f, indent=2)
    
    # Cost optimization recommendations
    cost_recommendations = cost_manager.generate_cost_optimization_recommendations()
    with open("cost-optimization/recommendations.json", "w") as f:
        json.dump(cost_recommendations, f, indent=2)
    
    # Auto-scaling configurations
    aks_hpa = scaling_manager.generate_aks_hpa_config(scaling_config)
    with open("cost-optimization/autoscaling/aks-hpa.yaml", "w") as f:
        json.dump(aks_hpa, f, indent=2)
    
    vmss_autoscale = scaling_manager.generate_vmss_autoscale_settings(scaling_config)
    with open("cost-optimization/autoscaling/vmss-autoscale.json", "w") as f:
        json.dump(vmss_autoscale, f, indent=2)
    
    app_service_autoscale = scaling_manager.generate_app_service_autoscale_settings(scaling_config)
    with open("cost-optimization/autoscaling/app-service-autoscale.json", "w") as f:
        json.dump(app_service_autoscale, f, indent=2)
    
    # Resource optimization
    print("🔧 Setting up resource optimization...")
    resource_optimizer = ResourceOptimizationManager()
    
    utilization_queries = resource_optimizer.generate_resource_utilization_analysis()
    with open("cost-optimization/monitoring/utilization-queries.json", "w") as f:
        json.dump(utilization_queries, f, indent=2)
    
    optimization_policies = resource_optimizer.generate_cost_optimization_policies()
    with open("cost-optimization/policies/optimization-policies.json", "w") as f:
        json.dump(optimization_policies, f, indent=2)
    
    # Performance vs cost optimization
    print("⚖️ Analyzing performance vs cost trade-offs...")
    perf_cost_optimizer = PerformanceCostOptimizer()
    
    perf_cost_analysis = perf_cost_optimizer.analyze_performance_cost_metrics()
    with open("cost-optimization/performance-cost-analysis.json", "w") as f:
        json.dump(perf_cost_analysis, f, indent=2)
    
    optimization_dashboard = perf_cost_optimizer.generate_optimization_dashboard()
    with open("cost-optimization/monitoring/optimization-dashboard.json", "w") as f:
        json.dump(optimization_dashboard, f, indent=2)
    
    # Spot instance management
    print("💸 Setting up Spot instance management...")
    spot_manager = SpotInstanceManager()
    
    spot_vmss = spot_manager.generate_spot_vm_scale_set()
    with open("cost-optimization/spot-instances/spot-vmss.json", "w") as f:
        json.dump(spot_vmss, f, indent=2)
    
    spot_monitoring = spot_manager.generate_spot_instance_monitoring()
    with open("cost-optimization/monitoring/spot-monitoring-queries.json", "w") as f:
        json.dump(spot_monitoring, f, indent=2)
    
    print("\n✅ Cost optimization and auto-scaling setup completed!")
    print("\n📁 Generated Files:")
    print("├── cost-optimization/")
    print("│   ├── budgets/")
    print("│   │   └── budget-alert.json")
    print("│   ├── autoscaling/")
    print("│   │   ├── aks-hpa.yaml")
    print("│   │   ├── vmss-autoscale.json")
    print("│   │   └── app-service-autoscale.json")
    print("│   ├── monitoring/")
    print("│   │   ├── cost-queries.json")
    print("│   │   ├── utilization-queries.json")
    print("│   │   ├── optimization-dashboard.json")
    print("│   │   └── spot-monitoring-queries.json")
    print("│   ├── policies/")
    print("│   │   └── optimization-policies.json")
    print("│   ├── spot-instances/")
    print("│   │   └── spot-vmss.json")
    print("│   ├── recommendations.json")
    print("│   └── performance-cost-analysis.json")
    
    print("\n🚀 Next Steps:")
    print("1. Deploy budget alerts and cost monitoring")
    print("2. Configure auto-scaling policies for your resources")
    print("3. Implement resource optimization recommendations")
    print("4. Set up Spot instances for batch workloads")
    print("5. Monitor performance vs cost metrics")
    print("6. Regular review and optimization of costs")
    
    print("\n💡 Cost Optimization Tips:")
    print("• Use GPT-3.5-turbo for 80% of queries, GPT-4 for complex reasoning")
    print("• Implement caching to reduce API calls by 40-70%")
    print("• Use Spot instances for non-critical workloads (60-90% savings)")
    print("• Auto-shutdown non-production resources during off-hours")
    print("• Right-size VMs based on actual utilization")
    print("• Implement storage lifecycle policies for long-term data")


if __name__ == "__main__":
    asyncio.run(demonstrate_cost_optimization())
