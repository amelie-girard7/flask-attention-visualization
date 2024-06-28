import os

class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///:memory:')  # This can be left as is or removed if not used
    DATA_PATH = os.getenv('DATA_PATH', 'data/model_2024-03-22-10/test_data_sample-attention.csv')
    ATTENTION_PATH = os.getenv('ATTENTION_PATH', 'data/model_2024-03-22-10/attentions')

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
