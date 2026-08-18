from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


class LocalStartupTest(TestCase):
    def test_startup_seeds_demo_login_and_allows_vite_origins(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "app.db"
            with patch.object(settings, "sqlite_path", database_path):
                with TestClient(app) as client:
                    health = client.get("/api/health")
                    self.assertEqual(health.status_code, 200)
                    self.assertTrue(health.json()["ok"])

                    login = client.post(
                        "/api/auth/login",
                        json={
                            "email": settings.demo_email,
                            "password": settings.demo_password,
                        },
                    )
                    self.assertEqual(login.status_code, 200)
                    self.assertEqual(login.json()["user"]["email"], settings.demo_email)

                    for origin in (
                        "http://127.0.0.1:5173",
                        "http://localhost:5173",
                        "http://127.0.0.1:4173",
                        "http://localhost:4173",
                    ):
                        with self.subTest(origin=origin):
                            preflight = client.options(
                                "/api/auth/login",
                                headers={
                                    "Origin": origin,
                                    "Access-Control-Request-Method": "POST",
                                    "Access-Control-Request-Headers": "content-type",
                                },
                            )
                            self.assertEqual(preflight.status_code, 200)
                            self.assertEqual(
                                preflight.headers.get("access-control-allow-origin"),
                                origin,
                            )
