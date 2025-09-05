"""
Azure Monitor and Application Insights Integration for LangChain Applications

This module provides comprehensive monitoring, logging, and alerting solutions
for LangChain applications deployed on Azure. It includes Azure Monitor integration,
Application Insights telemetry, custom metrics, alerting rules, and dashboard automation.

Key features:
1. Application Insights integration with OpenTelemetry
2. Custom metrics for LangChain operations
3. Distributed tracing for multi-component applications
4. Azure Monitor alert rules and action groups
5. Automated dashboard creation
6. Log Analytics workspace configuration
7. Performance monitoring and optimization
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from azure.identity import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
import opentelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
import time
import asyncio
from functools import wraps


# ============================================================================
# 1. AZURE MONITOR CONFIGURATION
# ============================================================================

@dataclass
class MonitoringConfig:
    """Azure Monitor configuration"""
    connection_string: str
    workspace_id: str
    workspace_key: str
    resource_group: str
    subscription_id: str
    application_name: str
    environment: str = "production"
    enable_telemetry: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    sample_rate: float = 1.0
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {
                "Application": self.application_name,
                "Environment": self.environment,
                "ManagedBy": "LangChain"
            }


class AzureMonitorSetup:
    """Set up Azure Monitor integration"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.tracer = None
        self.meter = None
        self.logger = None
        
    def initialize_monitoring(self):
        """Initialize Azure Monitor with OpenTelemetry"""
        try:
            # Configure Azure Monitor
            configure_azure_monitor(
                connection_string=self.config.connection_string,
                disable_offline_storage=False,
                enable_live_metrics=True,
                sampling_ratio=self.config.sample_rate
            )
            
            # Set up tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Set up meter for custom metrics
            self.meter = metrics.get_meter(__name__)
            
            # Set up logger
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Add custom properties to all telemetry
            self._add_custom_properties()
            
            print(f"✅ Azure Monitor initialized for {self.config.application_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Azure Monitor: {e}")
            raise
    
    def _add_custom_properties(self):
        """Add custom properties to telemetry"""
        # Add global properties that will be included in all telemetry
        from opentelemetry.sdk.resources import Resource
        
        resource = Resource.create({
            "service.name": self.config.application_name,
            "service.version": "1.0.0",
            "service.namespace": "langchain",
            "deployment.environment": self.config.environment
        })
        
        # Apply to tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)


# ============================================================================
# 2. CUSTOM METRICS FOR LANGCHAIN
# ============================================================================

class LangChainMetrics:
    """Custom metrics for LangChain operations"""
    
    def __init__(self, meter):
        self.meter = meter
        
        # Counter metrics
        self.llm_requests = self.meter.create_counter(
            name="langchain_llm_requests_total",
            description="Total number of LLM requests",
            unit="1"
        )
        
        self.chain_executions = self.meter.create_counter(
            name="langchain_chain_executions_total",
            description="Total number of chain executions",
            unit="1"
        )
        
        self.vector_store_operations = self.meter.create_counter(
            name="langchain_vector_store_operations_total",
            description="Total number of vector store operations",
            unit="1"
        )
        
        self.errors = self.meter.create_counter(
            name="langchain_errors_total",
            description="Total number of errors",
            unit="1"
        )
        
        # Histogram metrics
        self.llm_duration = self.meter.create_histogram(
            name="langchain_llm_duration_seconds",
            description="Duration of LLM requests",
            unit="s"
        )
        
        self.chain_duration = self.meter.create_histogram(
            name="langchain_chain_duration_seconds",
            description="Duration of chain executions",
            unit="s"
        )
        
        self.token_usage = self.meter.create_histogram(
            name="langchain_token_usage",
            description="Token usage per request",
            unit="1"
        )
        
        # Gauge metrics
        self.active_sessions = self.meter.create_up_down_counter(
            name="langchain_active_sessions",
            description="Number of active user sessions",
            unit="1"
        )
        
        self.cache_hit_rate = self.meter.create_gauge(
            name="langchain_cache_hit_rate",
            description="Cache hit rate percentage",
            unit="%"
        )
    
    def record_llm_request(self, model: str, duration: float, tokens: int, status: str):
        """Record LLM request metrics"""
        self.llm_requests.add(1, {"model": model, "status": status})
        self.llm_duration.record(duration, {"model": model})
        self.token_usage.record(tokens, {"model": model, "type": "total"})
    
    def record_chain_execution(self, chain_type: str, duration: float, status: str):
        """Record chain execution metrics"""
        self.chain_executions.add(1, {"chain_type": chain_type, "status": status})
        self.chain_duration.record(duration, {"chain_type": chain_type})
    
    def record_vector_operation(self, operation: str, collection: str, status: str):
        """Record vector store operation metrics"""
        self.vector_store_operations.add(1, {
            "operation": operation,
            "collection": collection,
            "status": status
        })
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        self.errors.add(1, {"error_type": error_type, "component": component})
    
    def update_active_sessions(self, count: int):
        """Update active sessions count"""
        self.active_sessions.add(count)
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate"""
        self.cache_hit_rate.set(hit_rate)


# ============================================================================
# 3. DISTRIBUTED TRACING
# ============================================================================

class LangChainTracer:
    """Distributed tracing for LangChain operations"""
    
    def __init__(self, tracer):
        self.tracer = tracer
    
    def trace_llm_call(self, model: str, prompt: str):
        """Trace LLM calls"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "llm_call",
                    attributes={
                        "llm.model": model,
                        "llm.prompt_length": len(prompt),
                        "llm.provider": "azure_openai"
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("llm.response_length", len(str(result)))
                        span.set_attribute("llm.status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("llm.status", "error")
                        span.set_attribute("llm.error", str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("llm.duration", duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "llm_call",
                    attributes={
                        "llm.model": model,
                        "llm.prompt_length": len(prompt),
                        "llm.provider": "azure_openai"
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("llm.response_length", len(str(result)))
                        span.set_attribute("llm.status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("llm.status", "error")
                        span.set_attribute("llm.error", str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("llm.duration", duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def trace_chain_execution(self, chain_type: str):
        """Trace chain executions"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "chain_execution",
                    attributes={
                        "chain.type": chain_type,
                        "chain.input_keys": str(kwargs.keys())
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("chain.status", "success")
                        span.set_attribute("chain.output_keys", str(result.keys()) if isinstance(result, dict) else "non-dict")
                        return result
                    except Exception as e:
                        span.set_attribute("chain.status", "error")
                        span.set_attribute("chain.error", str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("chain.duration", duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "chain_execution",
                    attributes={
                        "chain.type": chain_type,
                        "chain.input_keys": str(kwargs.keys())
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("chain.status", "success")
                        span.set_attribute("chain.output_keys", str(result.keys()) if isinstance(result, dict) else "non-dict")
                        return result
                    except Exception as e:
                        span.set_attribute("chain.status", "error")
                        span.set_attribute("chain.error", str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("chain.duration", duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


# ============================================================================
# 4. ALERT RULES CONFIGURATION
# ============================================================================

class AlertRulesGenerator:
    """Generate Azure Monitor alert rules"""
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_id: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_id = workspace_id
    
    def generate_high_error_rate_alert(self) -> Dict[str, Any]:
        """Generate alert for high error rate"""
        return {
            "type": "Microsoft.Insights/scheduledQueryRules",
            "apiVersion": "2021-08-01",
            "name": "langchain-high-error-rate",
            "location": "global",
            "properties": {
                "displayName": "LangChain - High Error Rate",
                "description": "Alert when error rate exceeds 5% over 5 minutes",
                "severity": 2,
                "enabled": True,
                "evaluationFrequency": "PT1M",
                "windowSize": "PT5M",
                "criteria": {
                    "allOf": [
                        {
                            "query": f"""
                                let errorRate = customMetrics
                                | where name == "langchain_errors_total"
                                | summarize ErrorCount = sum(value) by bin(timestamp, 1m)
                                | join (
                                    customMetrics
                                    | where name in ("langchain_llm_requests_total", "langchain_chain_executions_total")
                                    | summarize TotalCount = sum(value) by bin(timestamp, 1m)
                                ) on timestamp
                                | extend ErrorRate = (ErrorCount * 100.0) / TotalCount;
                                errorRate
                                | where ErrorRate > 5
                            """,
                            "timeAggregation": "Average",
                            "metricMeasureColumn": "ErrorRate",
                            "threshold": 5,
                            "operator": "GreaterThan"
                        }
                    ]
                },
                "actions": {
                    "actionGroups": [
                        f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Insights/actionGroups/langchain-alerts"
                    ]
                }
            }
        }
    
    def generate_high_latency_alert(self) -> Dict[str, Any]:
        """Generate alert for high latency"""
        return {
            "type": "Microsoft.Insights/scheduledQueryRules",
            "apiVersion": "2021-08-01",
            "name": "langchain-high-latency",
            "location": "global",
            "properties": {
                "displayName": "LangChain - High Latency",
                "description": "Alert when 95th percentile latency exceeds 10 seconds",
                "severity": 2,
                "enabled": True,
                "evaluationFrequency": "PT1M",
                "windowSize": "PT5M",
                "criteria": {
                    "allOf": [
                        {
                            "query": f"""
                                customMetrics
                                | where name == "langchain_llm_duration_seconds"
                                | summarize P95Latency = percentile(value, 95) by bin(timestamp, 1m)
                                | where P95Latency > 10
                            """,
                            "timeAggregation": "Average",
                            "metricMeasureColumn": "P95Latency",
                            "threshold": 10,
                            "operator": "GreaterThan"
                        }
                    ]
                },
                "actions": {
                    "actionGroups": [
                        f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Insights/actionGroups/langchain-alerts"
                    ]
                }
            }
        }
    
    def generate_token_usage_alert(self) -> Dict[str, Any]:
        """Generate alert for high token usage"""
        return {
            "type": "Microsoft.Insights/scheduledQueryRules",
            "apiVersion": "2021-08-01",
            "name": "langchain-high-token-usage",
            "location": "global",
            "properties": {
                "displayName": "LangChain - High Token Usage",
                "description": "Alert when token usage exceeds threshold",
                "severity": 1,
                "enabled": True,
                "evaluationFrequency": "PT5M",
                "windowSize": "PT15M",
                "criteria": {
                    "allOf": [
                        {
                            "query": f"""
                                customMetrics
                                | where name == "langchain_token_usage"
                                | summarize TotalTokens = sum(value) by bin(timestamp, 5m)
                                | where TotalTokens > 100000
                            """,
                            "timeAggregation": "Total",
                            "metricMeasureColumn": "TotalTokens",
                            "threshold": 100000,
                            "operator": "GreaterThan"
                        }
                    ]
                },
                "actions": {
                    "actionGroups": [
                        f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Insights/actionGroups/langchain-alerts"
                    ]
                }
            }
        }
    
    def generate_action_group(self) -> Dict[str, Any]:
        """Generate action group for alert notifications"""
        return {
            "type": "Microsoft.Insights/actionGroups",
            "apiVersion": "2023-01-01",
            "name": "langchain-alerts",
            "location": "global",
            "properties": {
                "groupShortName": "LCAlerts",
                "enabled": True,
                "emailReceivers": [
                    {
                        "name": "AdminEmail",
                        "emailAddress": "admin@company.com",
                        "useCommonAlertSchema": True
                    }
                ],
                "smsReceivers": [],
                "webhookReceivers": [
                    {
                        "name": "SlackWebhook",
                        "serviceUri": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                        "useCommonAlertSchema": True
                    }
                ],
                "armRoleReceivers": [
                    {
                        "name": "ContributorRole",
                        "roleId": "b24988ac-6180-42a0-ab88-20f7382dd24c",
                        "useCommonAlertSchema": True
                    }
                ]
            }
        }


# ============================================================================
# 5. DASHBOARD AUTOMATION
# ============================================================================

class DashboardGenerator:
    """Generate Azure Monitor dashboards"""
    
    def generate_langchain_dashboard(self, subscription_id: str, resource_group: str) -> Dict[str, Any]:
        """Generate comprehensive LangChain dashboard"""
        return {
            "type": "Microsoft.Portal/dashboards",
            "apiVersion": "2020-09-01-preview",
            "name": "langchain-monitoring-dashboard",
            "location": "global",
            "properties": {
                "lenses": [
                    {
                        "order": 0,
                        "parts": [
                            {
                                "position": {"x": 0, "y": 0, "rowSpan": 4, "colSpan": 6},
                                "metadata": {
                                    "inputs": [{
                                        "name": "query",
                                        "value": """
                                            customMetrics
                                            | where name == "langchain_llm_requests_total"
                                            | summarize RequestCount = sum(value) by bin(timestamp, 5m), tostring(customDimensions.model)
                                            | render timechart
                                        """
                                    }],
                                    "type": "Extension/HubsExtension/PartType/MonitorChartPart",
                                    "settings": {
                                        "content": {
                                            "options": {
                                                "chart": {
                                                    "metrics": [{
                                                        "resourceMetadata": {
                                                            "id": f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Insights/components/langchain-ai"
                                                        }
                                                    }]
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            {
                                "position": {"x": 6, "y": 0, "rowSpan": 4, "colSpan": 6},
                                "metadata": {
                                    "inputs": [{
                                        "name": "query",
                                        "value": """
                                            customMetrics
                                            | where name == "langchain_llm_duration_seconds"
                                            | summarize P50=percentile(value, 50), P95=percentile(value, 95), P99=percentile(value, 99) by bin(timestamp, 5m)
                                            | render timechart
                                        """
                                    }],
                                    "type": "Extension/HubsExtension/PartType/MonitorChartPart"
                                }
                            },
                            {
                                "position": {"x": 0, "y": 4, "rowSpan": 4, "colSpan": 6},
                                "metadata": {
                                    "inputs": [{
                                        "name": "query",
                                        "value": """
                                            customMetrics
                                            | where name == "langchain_errors_total"
                                            | summarize ErrorCount = sum(value) by bin(timestamp, 5m), tostring(customDimensions.error_type)
                                            | render timechart
                                        """
                                    }],
                                    "type": "Extension/HubsExtension/PartType/MonitorChartPart"
                                }
                            },
                            {
                                "position": {"x": 6, "y": 4, "rowSpan": 4, "colSpan": 6},
                                "metadata": {
                                    "inputs": [{
                                        "name": "query",
                                        "value": """
                                            customMetrics
                                            | where name == "langchain_token_usage"
                                            | summarize TotalTokens = sum(value) by bin(timestamp, 5m)
                                            | render timechart
                                        """
                                    }],
                                    "type": "Extension/HubsExtension/PartType/MonitorChartPart"
                                }
                            }
                        ]
                    }
                ],
                "metadata": {
                    "model": {
                        "timeRange": {
                            "value": {
                                "relative": {
                                    "duration": 24,
                                    "timeUnit": 1
                                }
                            },
                            "type": "MsPortalFx.Composition.Configuration.ValueTypes.TimeRange"
                        }
                    }
                }
            }
        }


# ============================================================================
# 6. PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Monitor LangChain application performance"""
    
    def __init__(self, metrics: LangChainMetrics, tracer: LangChainTracer):
        self.metrics = metrics
        self.tracer = tracer
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def monitor_llm_performance(self, func):
        """Decorator to monitor LLM performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            model = kwargs.get('model', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract token usage if available
                tokens = getattr(result, 'usage', {}).get('total_tokens', 0) if hasattr(result, 'usage') else 0
                
                # Record metrics
                self.metrics.record_llm_request(model, duration, tokens, "success")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                self.metrics.record_llm_request(model, duration, 0, "error")
                self.metrics.record_error(type(e).__name__, "llm")
                raise
        
        return wrapper
    
    def monitor_cache_performance(self, cache_key: str, cache_hit: bool):
        """Monitor cache performance"""
        if cache_hit:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1
        
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total > 0:
            hit_rate = (self.cache_stats["hits"] / total) * 100
            self.metrics.update_cache_hit_rate(hit_rate)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_cache_requests": total_requests,
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# 7. DEMONSTRATION AND USAGE
# ============================================================================

async def demonstrate_monitoring():
    """Demonstrate monitoring setup and usage"""
    print("📊 Azure Monitor and Application Insights Demo")
    print("=" * 60)
    
    # Configuration
    config = MonitoringConfig(
        connection_string="InstrumentationKey=your-key;IngestionEndpoint=https://your-region.in.applicationinsights.azure.com/",
        workspace_id="your-workspace-id",
        workspace_key="your-workspace-key",
        resource_group="your-rg",
        subscription_id="your-subscription-id",
        application_name="langchain-ai-app"
    )
    
    print("🔧 Initializing Azure Monitor...")
    monitor_setup = AzureMonitorSetup(config)
    monitor_setup.initialize_monitoring()
    
    # Set up custom metrics and tracing
    metrics = LangChainMetrics(monitor_setup.meter)
    tracer = LangChainTracer(monitor_setup.tracer)
    
    print("📈 Setting up performance monitoring...")
    performance_monitor = PerformanceMonitor(metrics, tracer)
    
    # Simulate some LangChain operations
    print("🔄 Simulating LangChain operations...")
    
    # Simulate LLM request
    @tracer.trace_llm_call("gpt-4", "What is the capital of France?")
    async def mock_llm_call():
        await asyncio.sleep(0.1)  # Simulate API call
        return {"response": "Paris", "usage": {"total_tokens": 25}}
    
    # Simulate chain execution
    @tracer.trace_chain_execution("qa_chain")
    async def mock_chain_execution():
        await asyncio.sleep(0.2)  # Simulate processing
        return {"answer": "Paris is the capital of France"}
    
    # Execute operations and record metrics
    for i in range(5):
        try:
            # LLM call
            result = await mock_llm_call()
            metrics.record_llm_request("gpt-4", 0.1, 25, "success")
            
            # Chain execution
            chain_result = await mock_chain_execution()
            metrics.record_chain_execution("qa_chain", 0.2, "success")
            
            # Vector store operation
            metrics.record_vector_operation("search", "documents", "success")
            
            # Cache simulation
            performance_monitor.monitor_cache_performance(f"cache_key_{i}", i % 3 == 0)
            
        except Exception as e:
            metrics.record_error(type(e).__name__, "simulation")
    
    print("📋 Generating alert rules...")
    alert_gen = AlertRulesGenerator(
        config.subscription_id, 
        config.resource_group, 
        config.workspace_id
    )
    
    # Generate ARM templates for alerts
    os.makedirs("monitoring/alerts", exist_ok=True)
    
    alert_rules = {
        "high-error-rate.json": alert_gen.generate_high_error_rate_alert(),
        "high-latency.json": alert_gen.generate_high_latency_alert(),
        "high-token-usage.json": alert_gen.generate_token_usage_alert(),
        "action-group.json": alert_gen.generate_action_group()
    }
    
    for filename, rule in alert_rules.items():
        with open(f"monitoring/alerts/{filename}", "w") as f:
            json.dump(rule, f, indent=2)
    
    print("📊 Generating dashboard...")
    dashboard_gen = DashboardGenerator()
    dashboard = dashboard_gen.generate_langchain_dashboard(
        config.subscription_id, 
        config.resource_group
    )
    
    os.makedirs("monitoring/dashboards", exist_ok=True)
    with open("monitoring/dashboards/langchain-dashboard.json", "w") as f:
        json.dump(dashboard, f, indent=2)
    
    print("📈 Performance Summary:")
    summary = performance_monitor.get_performance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Monitoring demonstration completed!")
    print("\n📁 Generated Files:")
    print("├── monitoring/")
    print("│   ├── alerts/")
    print("│   │   ├── high-error-rate.json")
    print("│   │   ├── high-latency.json")
    print("│   │   ├── high-token-usage.json")
    print("│   │   └── action-group.json")
    print("│   └── dashboards/")
    print("│       └── langchain-dashboard.json")
    
    print("\n🚀 Next Steps:")
    print("1. Deploy alert rules using Azure CLI or ARM templates")
    print("2. Configure Application Insights connection string")
    print("3. Set up notification channels (email, Slack, etc.)")
    print("4. Deploy dashboard to Azure Portal")
    print("5. Configure log retention and cost management")


# Sample usage in a real application
class MonitoredLangChainApp:
    """Example of a monitored LangChain application"""
    
    def __init__(self, monitoring_config: MonitoringConfig):
        self.monitor_setup = AzureMonitorSetup(monitoring_config)
        self.monitor_setup.initialize_monitoring()
        
        self.metrics = LangChainMetrics(self.monitor_setup.meter)
        self.tracer = LangChainTracer(self.monitor_setup.tracer)
        self.performance_monitor = PerformanceMonitor(self.metrics, self.tracer)
    
    @property
    def monitored_llm_call(self):
        """Get monitored LLM call decorator"""
        return self.performance_monitor.monitor_llm_performance
    
    @property  
    def traced_chain_execution(self):
        """Get traced chain execution decorator"""
        return self.tracer.trace_chain_execution


if __name__ == "__main__":
    asyncio.run(demonstrate_monitoring())
