from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    log_path: str
    diabetes_dataset_path: str
    heart_disease_dataset_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"