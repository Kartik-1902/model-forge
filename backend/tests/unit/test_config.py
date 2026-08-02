# pyrefly: ignore [missing-import]
from app.config import Settings, get_settings

def test_settings_load_defaults():
    settings = Settings()
    assert settings.app_name == "Model Forge"
    assert settings.app_version == "0.1.0"
    assert settings.environment in ["development", "testing", "production"]

def test_get_settings_caching():
    settings_1 = get_settings()
    settings_2 = get_settings()
    assert settings_1 is settings_2
