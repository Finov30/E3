-- Créer les bases de données
CREATE DATABASE IF NOT EXISTS `fastapi_db`;
CREATE DATABASE IF NOT EXISTS `faker_db`;

-- Créer l'utilisateur root s'il n'existe pas
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY 'root';

-- Donner tous les privilèges à root sur toutes les bases de données
GRANT ALL PRIVILEGES ON `fastapi_db`.* TO 'root'@'%';
GRANT ALL PRIVILEGES ON `faker_db`.* TO 'root'@'%';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;

-- Appliquer les changements de privilèges
FLUSH PRIVILEGES; 