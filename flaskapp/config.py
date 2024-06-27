import os

class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///:memory:')
    DATA_PATH = os.getenv('DATA_PATH', 'data/model_2024-03-22-10/test_data_sample-attention.csv')
    ATTENTION_PATH = os.getenv('ATTENTION_PATH', 'data/model_2024-03-22-10/attentions')

class ProductionConfig(Config):
    DATABASE_URI = os.getenv('DATABASE_URI')

class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///dev.db')

class TestingConfig(Config):
    TESTING = True
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///test.db')
