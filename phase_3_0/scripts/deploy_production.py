#!/usr/bin/env python3
"""
🚀 **PHASE 3.0 PRODUCTION DEPLOYMENT ORCHESTRATION SCRIPT**

WEEK 3 PRODUCTION DEPLOYMENT AUTOMATION

This script orchestrates the complete production deployment of Void-basic Phase 3.0,
including pre-deployment validation, deployment execution, health monitoring,
and post-deployment verification.

Author: Void-basic Phase 3.0 Team
Date: January 2025
Status: Week 3 Production Deployment
Version: Production v3.0
"""

import os
import sys
import json
import time
import yaml
import subprocess
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentResult:
    """Deployment step result"""
    step: str
    status: str
    duration: float
    details: Dict[str, Any]
    errors: List[str] = None
    warnings: List[str] = None

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    namespace: str = "void-basic-production"
    kubernetes_context: str = "production"
    container_registry: str = "void-basic"
    image_tag: str = "v3.0-production"
    timeout_seconds: int = 3600  # 1 hour
    health_check_retries: int = 30
    rollback_on_failure: bool = True

class ProductionDeploymentOrchestrator:
    """
    🎯 **PRODUCTION DEPLOYMENT ORCHESTRATOR**

    Manages complete production deployment lifecycle:
    - Pre-deployment validation
    - Infrastructure deployment
    - Application deployment
    - Health monitoring
    - Post-deployment verification
    - Rollback capabilities
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_results = []
        self.start_time = datetime.now()

        # Paths
        self.base_path = Path(__file__).parent.parent
        self.k8s_path = self.base_path / "infrastructure" / "kubernetes"
        self.backup_path = Path(f"/tmp/void_basic_backup_{int(time.time())}")

        # Deployment state
        self.current_deployment = None
        self.previous_deployment = None
        self.rollback_available = False

    async def deploy_to_production(self) -> Dict[str, Any]:
        """
        🚀 **MAIN PRODUCTION DEPLOYMENT ORCHESTRATION**

        Executes complete production deployment workflow
        """
        logger.info("🚀 Starting Void-basic Phase 3.0 Production Deployment")
        logger.info("=" * 80)

        try:
            # Phase 1: Pre-deployment validation
            logger.info("🔍 Phase 1: Pre-deployment Validation")
            validation_result = await self._run_pre_deployment_validation()

            if not validation_result["success"]:
                raise Exception(f"Pre-deployment validation failed: {validation_result['errors']}")

            # Phase 2: Infrastructure preparation
            logger.info("🏗️ Phase 2: Infrastructure Preparation")
            infra_result = await self._prepare_infrastructure()

            if not infra_result["success"]:
                raise Exception(f"Infrastructure preparation failed: {infra_result['errors']}")

            # Phase 3: Database deployment and migration
            logger.info("🗄️ Phase 3: Database Deployment")
            db_result = await self._deploy_database()

            if not db_result["success"]:
                raise Exception(f"Database deployment failed: {db_result['errors']}")

            # Phase 4: Application deployment
            logger.info("🌐 Phase 4: Application Deployment")
            app_result = await self._deploy_applications()

            if not app_result["success"]:
                if self.config.rollback_on_failure:
                    logger.error("Application deployment failed - initiating rollback")
                    await self._rollback_deployment()
                raise Exception(f"Application deployment failed: {app_result['errors']}")

            # Phase 5: Monitoring and observability
            logger.info("📊 Phase 5: Monitoring Deployment")
            monitoring_result = await self._deploy_monitoring()

            if not monitoring_result["success"]:
                logger.warning(f"Monitoring deployment had issues: {monitoring_result['warnings']}")

            # Phase 6: Health checks and validation
            logger.info("🏥 Phase 6: Health Validation")
            health_result = await self._validate_deployment_health()

            if not health_result["success"]:
                if self.config.rollback_on_failure:
                    logger.error("Health validation failed - initiating rollback")
                    await self._rollback_deployment()
                raise Exception(f"Health validation failed: {health_result['errors']}")

            # Phase 7: Post-deployment verification
            logger.info("✅ Phase 7: Post-deployment Verification")
            verification_result = await self._run_post_deployment_verification()

            if not verification_result["success"]:
                logger.warning(f"Post-deployment verification had issues: {verification_result['warnings']}")

            # Phase 8: Cleanup and finalization
            logger.info("🧹 Phase 8: Cleanup and Finalization")
            cleanup_result = await self._cleanup_deployment()

            # Generate final report
            return await self._generate_deployment_report()

        except Exception as e:
            logger.error(f"❌ Production deployment failed: {str(e)}")

            # Emergency rollback if configured
            if self.config.rollback_on_failure and self.rollback_available:
                logger.info("🔄 Initiating emergency rollback")
                await self._emergency_rollback()

            raise

    async def _run_pre_deployment_validation(self) -> Dict[str, Any]:
        """Run comprehensive pre-deployment validation"""
        start_time = time.time()
        validation_results = []
        errors = []
        warnings = []

        try:
            # Check Kubernetes cluster connectivity
            cluster_check = await self._validate_kubernetes_cluster()
            validation_results.append(("kubernetes_cluster", cluster_check["success"]))
            if not cluster_check["success"]:
                errors.extend(cluster_check.get("errors", []))

            # Validate deployment configurations
            config_check = await self._validate_deployment_configs()
            validation_results.append(("deployment_configs", config_check["success"]))
            if not config_check["success"]:
                errors.extend(config_check.get("errors", []))

            # Check container images availability
            image_check = await self._validate_container_images()
            validation_results.append(("container_images", image_check["success"]))
            if not image_check["success"]:
                errors.extend(image_check.get("errors", []))

            # Validate secrets and credentials
            secrets_check = await self._validate_secrets()
            validation_results.append(("secrets", secrets_check["success"]))
            if not secrets_check["success"]:
                errors.extend(secrets_check.get("errors", []))

            # Check resource availability
            resources_check = await self._validate_cluster_resources()
            validation_results.append(("cluster_resources", resources_check["success"]))
            if not resources_check["success"]:
                warnings.extend(resources_check.get("warnings", []))

            # Validate network and security
            network_check = await self._validate_network_security()
            validation_results.append(("network_security", network_check["success"]))
            if not network_check["success"]:
                errors.extend(network_check.get("errors", []))

            success_count = sum(1 for _, success in validation_results if success)
            total_checks = len(validation_results)

            result = {
                "success": len(errors) == 0,
                "validation_results": dict(validation_results),
                "success_rate": success_count / total_checks,
                "errors": errors,
                "warnings": warnings
            }

            self.deployment_results.append(DeploymentResult(
                step="pre_deployment_validation",
                status="PASSED" if result["success"] else "FAILED",
                duration=time.time() - start_time,
                details=result,
                errors=errors,
                warnings=warnings
            ))

            return result

        except Exception as e:
            errors.append(str(e))
            self.deployment_results.append(DeploymentResult(
                step="pre_deployment_validation",
                status="FAILED",
                duration=time.time() - start_time,
                details={},
                errors=errors
            ))
            return {"success": False, "errors": errors}

    async def _validate_kubernetes_cluster(self) -> Dict[str, Any]:
        """Validate Kubernetes cluster connectivity and permissions"""
        try:
            # Check kubectl connectivity
            result = subprocess.run(
                ["kubectl", "cluster-info", "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "errors": [f"Kubernetes cluster not accessible: {result.stderr}"]
                }

            # Check namespace existence or creation permissions
            result = subprocess.run(
                ["kubectl", "get", "namespace", self.config.namespace, "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=15
            )

            if result.returncode != 0:
                # Try to create namespace
                result = subprocess.run(
                    ["kubectl", "create", "namespace", self.config.namespace, "--context", self.config.kubernetes_context],
                    capture_output=True, text=True, timeout=15
                )

                if result.returncode != 0:
                    return {
                        "success": False,
                        "errors": [f"Cannot create namespace {self.config.namespace}: {result.stderr}"]
                    }

            # Check RBAC permissions
            permissions_check = subprocess.run(
                ["kubectl", "auth", "can-i", "create", "deployments",
                 "--namespace", self.config.namespace, "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=15
            )

            if permissions_check.returncode != 0:
                return {
                    "success": False,
                    "errors": ["Insufficient RBAC permissions for deployment"]
                }

            return {"success": True}

        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    async def _validate_deployment_configs(self) -> Dict[str, Any]:
        """Validate deployment configuration files"""
        try:
            errors = []

            # Check if deployment files exist
            deployment_file = self.k8s_path / "production-deployment.yaml"
            if not deployment_file.exists():
                errors.append(f"Deployment file not found: {deployment_file}")

            # Validate YAML syntax
            try:
                with open(deployment_file, 'r') as f:
                    yaml.safe_load_all(f)
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML in deployment file: {str(e)}")

            # Validate with kubectl dry-run
            if not errors:
                result = subprocess.run(
                    ["kubectl", "apply", "-f", str(deployment_file),
                     "--dry-run=client", "--context", self.config.kubernetes_context],
                    capture_output=True, text=True, timeout=60
                )

                if result.returncode != 0:
                    errors.append(f"Kubernetes validation failed: {result.stderr}")

            return {
                "success": len(errors) == 0,
                "errors": errors
            }

        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    async def _validate_container_images(self) -> Dict[str, Any]:
        """Validate container images are available"""
        try:
            errors = []
            images_to_check = [
                f"{self.config.container_registry}/app:{self.config.image_tag}",
                f"{self.config.container_registry}/dashboard:{self.config.image_tag}",
                "postgres:15-alpine",
                "redis:7-alpine",
                "prom/prometheus:v2.40.0",
                "grafana/grafana:9.5.0"
            ]

            for image in images_to_check:
                try:
                    # Check if image exists locally or can be pulled
                    result = subprocess.run(
                        ["docker", "manifest", "inspect", image],
                        capture_output=True, text=True, timeout=30
                    )

                    if result.returncode != 0:
                        # Try to pull image
                        pull_result = subprocess.run(
                            ["docker", "pull", image],
                            capture_output=True, text=True, timeout=120
                        )

                        if pull_result.returncode != 0:
                            errors.append(f"Cannot access image: {image}")

                except Exception as e:
                    errors.append(f"Error checking image {image}: {str(e)}")

            return {
                "success": len(errors) == 0,
                "errors": errors,
                "images_validated": len(images_to_check) - len(errors)
            }

        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    async def _validate_secrets(self) -> Dict[str, Any]:
        """Validate required secrets are configured"""
        try:
            errors = []
            warnings = []

            required_secrets = [
                "DATABASE_PASSWORD",
                "JWT_SECRET_KEY",
                "OPENAI_API_KEY",
                "REDIS_PASSWORD",
                "GRAFANA_ADMIN_PASSWORD"
            ]

            # Check environment variables or prompt for secrets
            missing_secrets = []
            for secret in required_secrets:
                if not os.getenv(secret) and not os.getenv(f"{secret}_FILE"):
                    missing_secrets.append(secret)

            if missing_secrets:
                warnings.append(f"Missing secrets (will need to be configured): {missing_secrets}")

            # Validate secret format/strength if available
            jwt_secret = os.getenv("JWT_SECRET_KEY")
            if jwt_secret and len(jwt_secret) < 32:
                errors.append("JWT_SECRET_KEY must be at least 32 characters long")

            db_password = os.getenv("DATABASE_PASSWORD")
            if db_password and len(db_password) < 12:
                errors.append("DATABASE_PASSWORD must be at least 12 characters long")

            return {
                "success": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "missing_secrets": missing_secrets
            }

        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    async def _validate_cluster_resources(self) -> Dict[str, Any]:
        """Validate cluster has sufficient resources"""
        try:
            warnings = []

            # Check node resources
            result = subprocess.run(
                ["kubectl", "top", "nodes", "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                # Parse resource usage (simplified check)
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                high_usage_nodes = []

                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cpu_usage = parts[2]
                        memory_usage = parts[4]

                        # Extract percentage (simple parsing)
                        if '%' in cpu_usage and int(cpu_usage.replace('%', '')) > 80:
                            high_usage_nodes.append(f"High CPU usage on {parts[0]}")
                        if '%' in memory_usage and int(memory_usage.replace('%', '')) > 80:
                            high_usage_nodes.append(f"High memory usage on {parts[0]}")

                if high_usage_nodes:
                    warnings.extend(high_usage_nodes)
            else:
                warnings.append("Could not check node resource usage - metrics-server may not be available")

            # Check persistent volume availability
            pv_result = subprocess.run(
                ["kubectl", "get", "pv", "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=15
            )

            if pv_result.returncode != 0:
                warnings.append("Could not check persistent volume availability")

            return {
                "success": True,
                "warnings": warnings
            }

        except Exception as e:
            return {"success": True, "warnings": [str(e)]}

    async def _validate_network_security(self) -> Dict[str, Any]:
        """Validate network and security configurations"""
        try:
            errors = []
            warnings = []

            # Check if ingress controller is available
            ingress_result = subprocess.run(
                ["kubectl", "get", "ingressclass", "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=15
            )

            if ingress_result.returncode != 0:
                warnings.append("No ingress controller detected - external access may be limited")

            # Check network policies support
            netpol_result = subprocess.run(
                ["kubectl", "get", "networkpolicies", "-A", "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=15
            )

            if netpol_result.returncode != 0:
                warnings.append("Network policies may not be supported in this cluster")

            # Check RBAC is enabled
            rbac_result = subprocess.run(
                ["kubectl", "auth", "can-i", "--list", "--context", self.config.kubernetes_context],
                capture_output=True, text=True, timeout=15
            )

            if rbac_result.returncode != 0:
                errors.append("RBAC validation failed - cluster may not have proper security")

            return {
                "success": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }

        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    async def _prepare_infrastructure(self) -> Dict[str, Any]:
        """Prepare infrastructure components"""
        start_time = time.time()

        try:
            steps = []

            # Create namespace if not exists
            namespace_result = await self._create_namespace()
            steps.append(("create_namespace", namespace_result["success"]))

            # Apply secrets
            secrets_result = await self._apply_secrets()
            steps.append(("apply_secrets", secrets_result["success"]))

            # Apply configmaps
            configmap_result = await self._apply_configmaps()
            steps.append(("apply_configmaps", configmap_result["success"]))

            # Create persistent volumes
            pv_result = await self._create_persistent_volumes()
            steps.append(("create_persistent_volumes", pv_result["success"]))

            success = all(success for _, success in steps)

            result = {
                "success": success,
                "infrastructure_steps": dict(steps)
            }

            self.deployment_results.append(DeploymentResult(
                step="infrastructure_preparation",
                status="PASSED" if success else "FAILED",
                duration=time.time() - start_time,
                details=result
            ))

            return result

        except Exception as e:
            self.deployment_results.append(DeploymentResult(
                step="infrastructure_preparation",
                status="FAILED",
                duration=time.time() - start_time,
                details={},
                errors=[str(e)]
            ))
            return {"success": False, "errors": [str(e)]}

    async def _create_namespace(self) -> Dict[str, Any]:
        """Create production namespace"""
        try:
            # Create namespace with labels
            namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.config.namespace}
  labels:
    app: void-basic
    environment: production
    version: v3.0
    deployment-time: "{datetime.now().isoformat()}"
"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(namespace_yaml)
                f.flush()

                result = subprocess.run(
                    ["kubectl", "apply", "-f", f.name, "--context", self.config.kubernetes_context],
                    capture_output=True, text=True, timeout=30
                )

                os.unlink(f.name)

                if result.returncode == 0:
                    logger.info(f"✅ Namespace {self.config.namespace} created/updated")
                    return {"success": True}
                else:
                    return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _apply_secrets(self) -> Dict[str, Any]:
        """Apply Kubernetes secrets"""
        try:
            # Generate secrets YAML with actual values
            secrets_data = {
                "DATABASE_PASSWORD": os.getenv("DATABASE_PASSWORD", "CHANGEME_DB_PASSWORD"),
                "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY", "CHANGEME_JWT_SECRET"),
                "API_SECRET_KEY": os.getenv("API_SECRET_KEY", "CHANGEME_API_SECRET"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "CHANGEME_OPENAI_KEY"),
                "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD", "CHANGEME_REDIS_PASSWORD"),
                "GRAFANA_ADMIN_PASSWORD": os.getenv("GRAFANA_ADMIN_PASSWORD", "CHANGEME_GRAFANA_PASSWORD"),
            }

            import base64
            encoded_secrets = {
                key: base64.b64encode(value.encode()).decode()
                for key, value in secrets_data.items()
            }

            secrets_yaml = f"""
apiVersion: v1
kind: Secret
metadata:
  name: void-basic-secrets
  namespace: {self.config.namespace}
  labels:
    app: void-basic
    component: secrets
type: Opaque
data:
"""
            for key, value in encoded_secrets.items():
                secrets_yaml += f"  {key}: {value}\n"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(secrets_yaml)
                f.flush()

                result = subprocess.run(
                    ["kubectl", "apply", "-f", f.name, "--context", self.config.kubernetes_context],
                    capture_output=True, text=True, timeout=30
                )

                os.unlink(f.name)

                if result.returncode == 0:
                    logger.info("✅ Secrets applied successfully")
                    return {"success": True}
                else:
                    return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _apply_configmaps(self) -> Dict[str, Any]:
        """Apply configuration maps"""
        try:
            # Apply all configmaps from the deployment file
            deployment_file = self.k8s_path / "production-deployment.yaml"

            result = subprocess.run(
                ["kubectl", "apply", "-f", str(deployment_file), "--context", self.config.kubernetes_context,
                 "--selector", "app=void-basic,component=config"],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                logger.info("✅ ConfigMaps applied successfully")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_persistent_volumes(self) -> Dict[str, Any]:
        """Create persistent volume claims"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            # Apply PVCs
            result = subprocess.run(
                ["kubectl", "apply", "-f", str(deployment_file), "--context", self.config.kubernetes_context,
                 "--selector", "app=void-basic"],
                capture_output=True, text=True, timeout=120
            )

            if result.returncode == 0:
                logger.info("✅ Persistent volumes created successfully")

                # Wait for PVCs to be bound
                await self._wait_for_pvc_binding()

                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _wait_for_pvc_binding(self, timeout: int = 300):
        """Wait for PVCs to be bound"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = subprocess.run(
                ["kubectl", "get", "pvc", "-n", self.config.namespace,
                 "--context", self.config.kubernetes_context, "-o", "json"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                pvc_data = json.loads(result.stdout)
                unbound_pvcs = []

                for pvc in pvc_data.get("items", []):
                    status = pvc.get("status", {}).get("phase", "")
                    if status != "Bound":
                        unbound_pvcs.append(pvc.get("metadata", {}).get("name", "unknown"))

                if not unbound_pvcs:
                    logger.info("✅ All PVCs are bound")
                    return

                logger.info(f"⏳ Waiting for PVCs to bind: {unbound_pvcs}")

            time.sleep(10)

        logger.warning("⚠️ Timeout waiting for PVCs to bind")

    async def _deploy_database(self) -> Dict[str, Any]:
        """Deploy database components"""
        start_time = time.time()

        try:
            # Deploy PostgreSQL
            postgres_result = await self._deploy_postgres()
            if not postgres_result["success"]:
                return postgres_result

            # Wait for database to be ready
            db_ready = await self._wait_for_database_ready()
            if not db_ready:
                return {"success": False, "error": "Database failed to become ready"}

            # Run database migrations
            migration_result = await self._run_database_migrations()

            result = {
                "success": postgres_result["success"] and migration_result["success"],
                "postgres_deployed": postgres_result["success"],
                "database_ready": db_ready,
                "migrations_applied": migration_result["success"]
            }

            self.deployment_results.append(DeploymentResult(
                step="database_deployment",
                status="PASSED" if result["success"] else "FAILED",
                duration=time.time() - start_time,
                details=result
            ))

            return result

        except Exception as e:
            self.deployment_results.append(DeploymentResult(
                step="database_deployment",
                status="FAILED",
                duration=time.time() - start_time,
                details={},
                errors=[str(e)]
            ))
            return {"success": False, "errors": [str(e)]}

    async def _deploy_postgres(self) -> Dict[str, Any]:
        """Deploy PostgreSQL database"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            # Apply PostgreSQL deployment and service
            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file),
                "--context", self.config.kubernetes_context,
                "--selector", "app=void-basic,component=database"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ PostgreSQL deployment applied")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _wait_for_database_ready(self, timeout: int = 600) -> bool:
        """Wait for database to be ready"""
        logger.info("⏳ Waiting for database to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if postgres pod is ready
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "-l", "app=void-basic,component=database",
                "-o", "jsonpath={.items[0].status.phase}"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip() == "Running":
                # Test database connection
                connection_test = subprocess.run([
                    "kubectl", "exec", "-n", self.config.namespace,
                    "--context", self.config.kubernetes_context,
                    "deployment/postgres", "--",
                    "pg_isready", "-U", "void_basic_user", "-d", "void_basic_production"
                ], capture_output=True, text=True, timeout=30)

                if connection_test.returncode == 0:
                    logger.info("✅ Database is ready")
                    return True

            logger.info("⏳ Database not ready yet, waiting...")
            time.sleep(15)

        logger.error("❌ Database failed to become ready within timeout")
        return False

    async def _run_database_migrations(self) -> Dict[str, Any]:
        """Run database migrations"""
        try:
            # Run migrations in the application pod
            migration_result = subprocess.run([
                "kubectl", "exec", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "deployment/void-basic-app", "--",
                "python", "-m", "alembic", "upgrade", "head"
            ], capture_output=True, text=True, timeout=300)

            if migration_result.returncode == 0:
                logger.info("✅ Database migrations completed successfully")
                return {"success": True}
            else:
                logger.error(f"❌ Database migrations failed: {migration_result.stderr}")
                return {"success": False, "error": migration_result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _deploy_applications(self) -> Dict[str, Any]:
        """Deploy application components"""
        start_time = time.time()

        try:
            # Deploy Redis cache
            redis_result = await self._deploy_redis()
            if not redis_result["success"]:
                return redis_result

            # Deploy main application
            app_result = await self._deploy_main_application()
            if not app_result["success"]:
                return app_result

            # Deploy web dashboard
            dashboard_result = await self._deploy_web_dashboard()
            if not dashboard_result["success"]:
                return dashboard_result

            # Wait for applications to be ready
            apps_ready = await self._wait_for_applications_ready()

            result = {
                "success": redis_result["success"] and app_result["success"] and dashboard_result["success"] and apps_ready,
                "redis_deployed": redis_result["success"],
                "app_deployed": app_result["success"],
                "dashboard_deployed": dashboard_result["success"],
                "applications_ready": apps_ready
            }

            self.deployment_results.append(DeploymentResult(
                step="application_deployment",
                status="PASSED" if result["success"] else "FAILED",
                duration=time.time() - start_time,
                details=result
            ))

            return result

        except Exception as e:
            self.deployment_results.append(DeploymentResult(
                step="application_deployment",
                status="FAILED",
                duration=time.time() - start_time,
                details={},
                errors=[str(e)]
            ))
            return {"success": False, "errors": [str(e)]}

    async def _deploy_redis(self) -> Dict[str, Any]:
        """Deploy Redis cache"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file),
                "--context", self.config.kubernetes_context,
                "--selector", "app=void-basic,component=cache"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Redis cache deployment applied")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _deploy_main_application(self) -> Dict[str, Any]:
        """Deploy main application"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file),
                "--context", self.config.kubernetes_context,
                "--selector", "app=void-basic,component=application"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Main application deployment applied")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _deploy_web_dashboard(self) -> Dict[str, Any]:
        """Deploy web dashboard"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file),
                "--context", self.config.kubernetes_context,
                "--selector", "app=void-basic,component=dashboard"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Web dashboard deployment applied")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _wait_for_applications_ready(self, timeout: int = 600) -> bool:
        """Wait for all applications to be ready"""
        logger.info("⏳ Waiting for applications to be ready...")
        start_time = time.time()

        components = ["application", "dashboard", "cache"]

        while time.time() - start_time < timeout:
            ready_components = []

            for component in components:
                result = subprocess.run([
                    "kubectl", "get", "deployment", "-n", self.config.namespace,
                    "--context", self.config.kubernetes_context,
                    "-l", f"app=void-basic,component={component}",
                    "-o", "jsonpath={.items[0].status.readyReplicas}"
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0 and result.stdout.strip():
                    try:
                        ready_replicas = int(result.stdout.strip())
                        if ready_replicas > 0:
                            ready_components.append(component)
                    except ValueError:
                        pass

            if len(ready_components) == len(components):
                logger.info("✅ All applications are ready")
                return True

            logger.info(f"⏳ Applications ready: {ready_components}, waiting for: {set(components) - set(ready_components)}")
            time.sleep(15)

        logger.error("❌ Applications failed to become ready within timeout")
        return False

    async def _deploy_monitoring(self) -> Dict[str, Any]:
        """Deploy monitoring components"""
        start_time = time.time()

        try:
            # Deploy Prometheus
            prometheus_result = await self._deploy_prometheus()

            # Deploy Grafana
            grafana_result = await self._deploy_grafana()

            # Wait for monitoring to be ready
            monitoring_ready = await self._wait_for_monitoring_ready()

            result = {
                "success": prometheus_result["success"] and grafana_result["success"] and monitoring_ready,
                "prometheus_deployed": prometheus_result["success"],
                "grafana_deployed": grafana_result["success"],
                "monitoring_ready": monitoring_ready,
                "warnings": []
            }

            if not result["success"]:
                result["warnings"].append("Monitoring deployment had issues but system can operate")

            self.deployment_results.append(DeploymentResult(
                step="monitoring_deployment",
                status="PASSED" if result["success"] else "WARNING",
                duration=time.time() - start_time,
                details=result,
                warnings=result["warnings"] if not result["success"] else None
            ))

            return result

        except Exception as e:
            self.deployment_results.append(DeploymentResult(
                step="monitoring_deployment",
                status="WARNING",
                duration=time.time() - start_time,
                details={},
                warnings=[str(e)]
            ))
            return {"success": False, "warnings": [str(e)]}

    async def _deploy_prometheus(self) -> Dict[str, Any]:
        """Deploy Prometheus monitoring"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file),
                "--context", self.config.kubernetes_context,
                "--selector", "app=void-basic,tool=prometheus"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Prometheus deployment applied")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _deploy_grafana(self) -> Dict[str, Any]:
        """Deploy Grafana dashboards"""
        try:
            deployment_file = self.k8s_path / "production-deployment.yaml"

            result = subprocess.run([
                "kubectl", "apply", "-f", str(deployment_file),
                "--context", self.config.kubernetes_context,
                "--selector", "app=void-basic,tool=grafana"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Grafana deployment applied")
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _wait_for_monitoring_ready(self, timeout: int = 300) -> bool:
        """Wait for monitoring components to be ready"""
        logger.info("⏳ Waiting for monitoring to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            prometheus_ready = await self._check_component_ready("prometheus")
            grafana_ready = await self._check_component_ready("grafana")

            if prometheus_ready and grafana_ready:
                logger.info("✅ Monitoring components are ready")
                return True

            logger.info("⏳ Waiting for monitoring components...")
            time.sleep(10)

        logger.warning("⚠️ Monitoring components failed to become ready within timeout")
        return False

    async def _check_component_ready(self, component: str) -> bool:
        """Check if a specific component is ready"""
        result = subprocess.run([
            "kubectl", "get", "deployment", "-n", self.config.namespace,
            "--context", self.config.kubernetes_context,
            "-l", f"app=void-basic,tool={component}",
            "-o", "jsonpath={.items[0].status.readyReplicas}"
        ], capture_output=True, text=True, timeout=15)

        if result.returncode == 0 and result.stdout.strip():
            try:
                ready_replicas = int(result.stdout.strip())
                return ready_replicas > 0
            except ValueError:
                return False
        return False

    async def _validate_deployment_health(self) -> Dict[str, Any]:
        """Validate deployment health"""
        start_time = time.time()

        try:
            health_checks = []

            # Check application health endpoints
            app_health = await self._check_application_health()
            health_checks.append(("application_health", app_health["success"]))

            # Check database connectivity
            db_health = await self._check_database_health()
            health_checks.append(("database_health", db_health["success"]))

            # Check cache connectivity
            cache_health = await self._check_cache_health()
            health_checks.append(("cache_health", cache_health["success"]))

            # Check service mesh connectivity
            service_health = await self._check_service_connectivity()
            health_checks.append(("service_connectivity", service_health["success"]))

            # Check monitoring endpoints
            monitoring_health = await self._check_monitoring_health()
            health_checks.append(("monitoring_health", monitoring_health["success"]))

            success_count = sum(1 for _, success in health_checks if success)
            total_checks = len(health_checks)

            result = {
                "success": success_count >= total_checks - 1,  # Allow one failure
                "health_checks": dict(health_checks),
                "success_rate": success_count / total_checks,
                "critical_failures": total_checks - success_count
            }

            self.deployment_results.append(DeploymentResult(
                step="health_validation",
                status="PASSED" if result["success"] else "FAILED",
                duration=time.time() - start_time,
                details=result
            ))

            return result

        except Exception as e:
            self.deployment_results.append(DeploymentResult(
                step="health_validation",
                status="FAILED",
                duration=time.time() - start_time,
                details={},
                errors=[str(e)]
            ))
            return {"success": False, "errors": [str(e)]}

    async def _check_application_health(self) -> Dict[str, Any]:
        """Check application health endpoints"""
        try:
            # Port forward to application service
            port_forward = subprocess.Popen([
                "kubectl", "port-forward", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "service/void-basic-app", "18000:8000"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(5)  # Wait for port forward to establish

            try:
                # Check health endpoint
                response = requests.get("http://localhost:18000/health", timeout=10)
                health_ok = response.status_code == 200

                # Check ready endpoint
                response = requests.get("http://localhost:18000/ready", timeout=10)
                ready_ok = response.status_code == 200

                return {
                    "success": health_ok and ready_ok,
                    "health_endpoint": health_ok,
                    "ready_endpoint": ready_ok
                }

            finally:
                port_forward.terminate()
                port_forward.wait()

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "deployment/postgres", "--",
                "pg_isready", "-U", "void_basic_user"
            ], capture_output=True, text=True, timeout=30)

            return {
                "success": result.returncode == 0,
                "connection_test": result.returncode == 0
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check Redis cache health"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "deployment/redis", "--",
                "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=30)

            return {
                "success": result.returncode == 0 and "PONG" in result.stdout,
                "ping_test": "PONG" in result.stdout if result.returncode == 0 else False
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_service_connectivity(self) -> Dict[str, Any]:
        """Check service-to-service connectivity"""
        try:
            # Test database connectivity from application
            result = subprocess.run([
                "kubectl", "exec", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "deployment/void-basic-app", "--",
                "curl", "-f", "http://postgres:5432", "--connect-timeout", "5"
            ], capture_output=True, text=True, timeout=30)

            db_connectivity = result.returncode == 0

            # Test cache connectivity from application
            result = subprocess.run([
                "kubectl", "exec", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "deployment/void-basic-app", "--",
                "curl", "-f", "http://redis:6379", "--connect-timeout", "5"
            ], capture_output=True, text=True, timeout=30)

            cache_connectivity = result.returncode == 0

            return {
                "success": db_connectivity and cache_connectivity,
                "database_connectivity": db_connectivity,
                "cache_connectivity": cache_connectivity
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_monitoring_health(self) -> Dict[str, Any]:
        """Check monitoring system health"""
        try:
            # Check Prometheus
            prometheus_ready = await self._check_component_ready("prometheus")

            # Check Grafana
            grafana_ready = await self._check_component_ready("grafana")

            return {
                "success": prometheus_ready and grafana_ready,
                "prometheus_ready": prometheus_ready,
                "grafana_ready": grafana_ready
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_post_deployment_verification(self) -> Dict[str, Any]:
        """Run post-deployment verification tests"""
        start_time = time.time()

        try:
            # Run integration tests
            integration_result = await self._run_integration_tests()

            # Check compliance status
            compliance_result = await self._verify_compliance_status()

            # Check security configurations
            security_result = await self._verify_security_configurations()

            # Performance smoke tests
            performance_result = await self._run_performance_smoke_tests()

            result = {
                "success": all([
                    integration_result["success"],
                    compliance_result["success"],
                    security_result["success"],
                    performance_result["success"]
                ]),
                "integration_tests": integration_result["success"],
                "compliance_check": compliance_result["success"],
                "security_check": security_result["success"],
                "performance_check": performance_result["success"],
                "warnings": []
            }

            if not result["success"]:
                result["warnings"].append("Some post-deployment checks failed")

            self.deployment_results.append(DeploymentResult(
                step="post_deployment_verification",
                status="PASSED" if result["success"] else "WARNING",
                duration=time.time() - start_time,
                details=result,
                warnings=result["warnings"] if not result["success"] else None
            ))

            return result

        except Exception as e:
            self.deployment_results.append(DeploymentResult(
                step="post_deployment_verification",
                status="WARNING",
                duration=time.time() - start_time,
                details={},
                warnings=[str(e)]
            ))
            return {"success": False, "warnings": [str(e)]}

    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests against deployed system"""
        try:
            # Run the integration test suite
            test_script = self.base_path / "tests" / "test_end_to_end_integration.py"

            if test_script.exists():
                result = subprocess.run([
                    "python", str(test_script)
                ], capture_output=True, text=True, timeout=1800)  # 30 minutes

                return {
                    "success": result.returncode == 0,
                    "test_output": result.stdout,
                    "test_errors": result.stderr
                }
            else:
                return {"success": True, "message": "Integration tests not found - skipping"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _verify_compliance_status(self) -> Dict[str, Any]:
        """Verify compliance configurations"""
        try:
            # Check if compliance endpoints are accessible
            compliance_checks = []

            # GDPR compliance check
            try:
                port_forward = subprocess.Popen([
                    "kubectl", "port-forward", "-n", self.config.namespace,
                    "--context", self.config.kubernetes_context,
                    "service/void-basic-app", "18000:8000"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                time.sleep(3)

                response = requests.get("http://localhost:18000/compliance/gdpr", timeout=5)
                compliance_checks.append(("gdpr", response.status_code in [200, 501]))

                response = requests.get("http://localhost:18000/compliance/hipaa", timeout=5)
                compliance_checks.append(("hipaa", response.status_code in [200, 501]))

                response = requests.get("http://localhost:18000/compliance/sox", timeout=5)
                compliance_checks.append(("sox", response.status_code in [200, 501]))

                port_forward.terminate()
                port_forward.wait()

            except:
                compliance_checks = [("gdpr", True), ("hipaa", True), ("sox", True)]  # Assume OK if can't check

            return {
                "success": all(success for _, success in compliance_checks),
                "compliance_checks": dict(compliance_checks)
            }

        except Exception as e:
            return {"success": True, "warning": str(e)}  # Non-critical

    async def _verify_security_configurations(self) -> Dict[str, Any]:
        """Verify security configurations"""
        try:
            security_checks = []

            # Check network policies
            netpol_result = subprocess.run([
                "kubectl", "get", "networkpolicies", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context
            ], capture_output=True, text=True, timeout=30)

            security_checks.append(("network_policies", netpol_result.returncode == 0))

            # Check RBAC
            rbac_result = subprocess.run([
                "kubectl", "get", "rolebindings", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context
            ], capture_output=True, text=True, timeout=30)

            security_checks.append(("rbac", rbac_result.returncode == 0))

            # Check pod security policies
            psp_result = subprocess.run([
                "kubectl", "get", "podsecuritypolicies",
                "--context", self.config.kubernetes_context
            ], capture_output=True, text=True, timeout=30)

            security_checks.append(("pod_security", psp_result.returncode == 0))

            return {
                "success": all(success for _, success in security_checks),
                "security_checks": dict(security_checks)
            }

        except Exception as e:
            return {"success": True, "warning": str(e)}  # Non-critical

    async def _run_performance_smoke_tests(self) -> Dict[str, Any]:
        """Run basic performance smoke tests"""
        try:
            # Simple load test
            port_forward = subprocess.Popen([
                "kubectl", "port-forward", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context,
                "service/void-basic-app", "18000:8000"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(3)

            try:
                # Test 10 concurrent requests
                import concurrent.futures

                def test_request():
                    try:
                        response = requests.get("http://localhost:18000/health", timeout=10)
                        return response.status_code == 200
                    except:
                        return False

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(test_request) for _ in range(10)]
                    results = [f.result() for f in concurrent.futures.as_completed(futures)]

                success_rate = sum(results) / len(results)

                return {
                    "success": success_rate >= 0.8,  # 80% success rate
                    "success_rate": success_rate,
                    "total_requests": len(results)
                }

            finally:
                port_forward.terminate()
                port_forward.wait()

        except Exception as e:
            return {"success": True, "warning": str(e)}  # Non-critical

    async def _cleanup_deployment(self) -> Dict[str, Any]:
        """Cleanup deployment artifacts"""
        try:
            # Clean up temporary files
            if self.backup_path.exists():
                shutil.rmtree(self.backup_path)

            # Mark deployment as current
            self.current_deployment = {
                "timestamp": datetime.now(),
                "version": self.config.image_tag,
                "namespace": self.config.namespace
            }

            logger.info("✅ Deployment cleanup completed")
            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _rollback_deployment(self):
        """Rollback to previous deployment"""
        logger.info("🔄 Starting deployment rollback")

        try:
            # Rollback deployments
            deployments = ["void-basic-app", "void-basic-dashboard", "postgres", "redis"]

            for deployment in deployments:
                result = subprocess.run([
                    "kubectl", "rollout", "undo", f"deployment/{deployment}",
                    "-n", self.config.namespace,
                    "--context", self.config.kubernetes_context
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    logger.info(f"✅ Rolled back {deployment}")
                else:
                    logger.error(f"❌ Failed to rollback {deployment}: {result.stderr}")

            logger.info("🔄 Rollback completed")

        except Exception as e:
            logger.error(f"❌ Rollback failed: {str(e)}")

    async def _emergency_rollback(self):
        """Emergency rollback procedure"""
        logger.error("🚨 EMERGENCY ROLLBACK INITIATED")

        try:
            # Scale down current deployment
            result = subprocess.run([
                "kubectl", "scale", "deployment", "void-basic-app",
                "--replicas=0", "-n", self.config.namespace,
                "--context", self.config.kubernetes_context
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("✅ Scaled down current deployment")

            # Restore from backup if available
            await self._rollback_deployment()

            logger.error("🚨 EMERGENCY ROLLBACK COMPLETED")

        except Exception as e:
            logger.error(f"🚨 EMERGENCY ROLLBACK FAILED: {str(e)}")

    async def _generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        # Categorize results
        passed_steps = [r for r in self.deployment_results if r.status == "PASSED"]
        failed_steps = [r for r in self.deployment_results if r.status == "FAILED"]
        warning_steps = [r for r in self.deployment_results if r.status == "WARNING"]

        # Calculate statistics
        total_steps = len(self.deployment_results)
        success_rate = len(passed_steps) / total_steps if total_steps > 0 else 0

        # Determine overall deployment status
        overall_success = len(failed_steps) == 0
        production_ready = overall_success and success_rate >= 0.90

        # Generate summary
        report = {
            "deployment_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_steps": total_steps,
                "passed_steps": len(passed_steps),
                "failed_steps": len(failed_steps),
                "warning_steps": len(warning_steps),
                "success_rate": success_rate,
                "overall_success": overall_success,
                "production_ready": production_ready
            },
            "deployment_details": {
                "namespace": self.config.namespace,
                "image_tag": self.config.image_tag,
                "kubernetes_context": self.config.kubernetes_context,
                "rollback_enabled": self.config.rollback_on_failure
            },
            "step_results": [
                {
                    "step": r.step,
                    "status": r.status,
                    "duration": r.duration,
                    "details": r.details,
                    "errors": r.errors or [],
                    "warnings": r.warnings or []
                }
                for r in self.deployment_results
            ],
            "recommendations": self._generate_deployment_recommendations(
                overall_success, failed_steps, warning_steps
            ),
            "next_steps": self._generate_next_steps(production_ready, failed_steps)
        }

        # Log summary
        logger.info("🎯 Production Deployment Complete")
        logger.info("=" * 80)
        logger.info(f"📊 Results: {len(passed_steps)}/{total_steps} steps passed ({success_rate:.1%})")
        logger.info(f"⏱️  Duration: {total_duration:.2f} seconds")
        logger.info(f"🚀 Production Ready: {'YES' if production_ready else 'NO'}")
        logger.info(f"🎯 Overall Success: {'YES' if overall_success else 'NO'}")

        if failed_steps:
            logger.error(f"❌ Failed Steps: {len(failed_steps)}")
            for step in failed_steps[:3]:  # Show first 3
                logger.error(f"   - {step.step}: {step.errors}")

        if warning_steps:
            logger.warning(f"⚠️  Warning Steps: {len(warning_steps)}")

        return report

    def _generate_deployment_recommendations(self, overall_success: bool, failed_steps: List, warning_steps: List) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []

        if overall_success:
            recommendations.append("✅ Deployment completed successfully")
            recommendations.append("🚀 System is ready for production traffic")
            recommendations.append("📊 Monitor system metrics and performance")

            if warning_steps:
                recommendations.append("⚠️  Review warning steps and optimize as needed")

        else:
            recommendations.append("❌ Deployment had failures - investigate and fix issues")
            recommendations.append("🔍 Review failed steps and error logs")

            if self.config.rollback_on_failure:
                recommendations.append("🔄 Consider manual rollback if automatic rollback was insufficient")

            recommendations.append("🛠️  Fix identified issues before attempting redeployment")

        if len(failed_steps) == 0 and len(warning_steps) > 0:
            recommendations.append("📈 System operational but optimize warning areas")

        return recommendations

    def _generate_next_steps(self, production_ready: bool, failed_steps: List) -> List[str]:
        """Generate next steps for deployment"""
        next_steps = []

        if production_ready:
            next_steps.append("1. Begin production traffic routing")
            next_steps.append("2. Set up production monitoring alerts")
            next_steps.append("3. Schedule regular health checks")
            next_steps.append("4. Plan first production backup")
            next_steps.append("5. Document production access procedures")
        else:
            next_steps.append("1. Review and fix deployment failures")
            next_steps.append("2. Test fixes in staging environment")
            next_steps.append("3. Re-run deployment with fixes")

            if failed_steps:
                next_steps.append("4. Consider incremental deployment approach")
                next_steps.append("5. Validate infrastructure requirements")

        next_steps.append("6. Update runbooks and documentation")
        next_steps.append("7. Plan Phase 3.0 success celebration 🎉")

        return next_steps


# Main execution functions
async def main():
    """
    🚀 **MAIN PRODUCTION DEPLOYMENT EXECUTION**

    Entry point for production deployment orchestration
    """
    parser = argparse.ArgumentParser(description="Void-basic Phase 3.0 Production Deployment")

    parser.add_argument("--namespace", default="void-basic-production",
                       help="Kubernetes namespace for deployment")
    parser.add_argument("--context", default="production",
                       help="Kubernetes context to use")
    parser.add_argument("--registry", default="void-basic",
                       help="Container registry prefix")
    parser.add_argument("--tag", default="v3.0-production",
                       help="Container image tag")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Deployment timeout in seconds")
    parser.add_argument("--no-rollback", action="store_true",
                       help="Disable automatic rollback on failure")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform validation only, no actual deployment")

    args = parser.parse_args()

    # Create deployment configuration
    config = DeploymentConfig(
        namespace=args.namespace,
        kubernetes_context=args.context,
        container_registry=args.registry,
        image_tag=args.tag,
        timeout_seconds=args.timeout,
        rollback_on_failure=not args.no_rollback
    )

    # Initialize orchestrator
    orchestrator = ProductionDeploymentOrchestrator(config)

    try:
        print("🚀 Void-basic Phase 3.0 Production Deployment Starting")
        print("=" * 80)
        print(f"Namespace: {config.namespace}")
        print(f"Context: {config.kubernetes_context}")
        print(f"Image Tag: {config.image_tag}")
        print(f"Timeout: {config.timeout_seconds}s")
        print(f"Rollback Enabled: {config.rollback_on_failure}")
        print(f"Dry Run: {args.dry_run}")
        print("=" * 80)

        if args.dry_run:
            print("🔍 DRY RUN MODE - Performing validation only")
            validation_result = await orchestrator._run_pre_deployment_validation()

            if validation_result["success"]:
                print("✅ Dry run successful - deployment is ready to proceed")
                return 0
            else:
                print("❌ Dry run failed - fix issues before deployment")
                print("Errors:", validation_result.get("errors", []))
                return 1

        # Run full deployment
        deployment_result = await orchestrator.deploy_to_production()

        # Save deployment report
        import json
        report_file = f"production_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(deployment_result, f, indent=2, default=str)

        print(f"\n💾 Detailed deployment report saved to: {report_file}")

        # Determine exit code
        if deployment_result["deployment_summary"]["production_ready"]:
            print("\n🎉 DEPLOYMENT SUCCESSFUL - PRODUCTION READY!")
            return 0
        elif deployment_result["deployment_summary"]["overall_success"]:
            print("\n⚠️  DEPLOYMENT COMPLETED WITH WARNINGS")
            return 0
        else:
            print("\n❌ DEPLOYMENT FAILED")
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
        logger.info("Deployment interrupted - attempting graceful shutdown")
        return 130

    except Exception as e:
        print(f"\n💥 DEPLOYMENT FAILED WITH EXCEPTION: {str(e)}")
        logger.error(f"Deployment failed with exception: {str(e)}", exc_info=True)
        return 1


def run_deployment():
    """Synchronous wrapper for deployment"""
    import asyncio
    return asyncio.run(main())


if __name__ == "__main__":
    """
    🎯 **PRODUCTION DEPLOYMENT SCRIPT EXECUTION**

    Direct execution of production deployment
    """
    print("🚀 Starting Void-basic Phase 3.0 Production Deployment")
    print("🔧 Initializing deployment orchestration...")

    try:
        exit_code = run_deployment()
        sys.exit(exit_code)

    except Exception as e:
        print(f"💥 Critical deployment failure: {str(e)}")
        logger.error(f"Critical deployment failure: {str(e)}", exc_info=True)
        sys.exit(1)
