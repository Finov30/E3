from sqlalchemy import Table, Column, Integer, String, ForeignKey
from config.database import meta

# Définition de la table users
users = Table(
    'users', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(50)),
    Column('email', String(50)),
    Column('password', String(50)),
    extend_existing=True
)

# Définition de la table address
address_table = Table(
    'address', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
    Column('street', String(255), nullable=False),
    Column('zipcode', String(20), nullable=False),
    Column('country', String(100), nullable=False)
)
