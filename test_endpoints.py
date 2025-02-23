import requests
import json
from datetime import datetime
import os

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.test_results = []
        
    def log_test(self, endpoint, method, status, success, request_data=None, response_data=None, error=None, headers=None, files=None):
        """Enregistre le résultat d'un test avec les détails complets de la requête"""
        # Préparer les informations sur les fichiers
        files_info = None
        if files:
            if isinstance(files, list):
                files_info = [{"filename": f[1].name} for f in files]  # Pour predict_batch
            else:
                files_info = {"filename": next(iter(files.values())).name}  # Pour predict et predict_zip

        result = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "request": {
                "url": f"{self.base_url}{endpoint}",
                "headers": headers,
                "data": request_data,
                "files": files_info
            },
            "response": {
                "status_code": status,
                "data": response_data
            },
            "success": success,
            "error": str(error) if error else None
        }
        self.test_results.append(result)
        return result

    def login(self, username, password):
        """Test de login et récupération du token"""
        endpoint = "/auth/login"
        try:
            print(f"\n🔑 Test de {endpoint}")
            data = {"username": username, "password": password}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                data=data
            )
            
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                print("✅ Login réussi")
                return self.log_test(
                    endpoint=endpoint,
                    method="POST",
                    status=response.status_code,
                    success=True,
                    request_data=data,
                    response_data=response.json()
                )
            else:
                print(f"❌ Login échoué: {response.status_code}")
                return self.log_test(
                    endpoint=endpoint,
                    method="POST",
                    status=response.status_code,
                    success=False,
                    request_data=data,
                    response_data=response.json()
                )
                
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return self.log_test(
                endpoint=endpoint,
                method="POST",
                status=None,
                success=False,
                request_data=data,
                error=e
            )

    def test_predict(self, image_path):
        """Test de l'endpoint predict"""
        endpoint = "/predict"
        try:
            print(f"\n🖼️ Test de {endpoint}")
            headers = {'Authorization': f'Bearer {self.token}'}
            with open(image_path, 'rb') as img:
                files = {'file': img}
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    files=files,
                    headers=headers
                )
            
            if response.status_code == 200:
                print("✅ Prédiction réussie")
                return self.log_test(
                    endpoint=endpoint,
                    method="POST",
                    status=response.status_code,
                    success=True,
                    request_data={"image_path": image_path},
                    response_data=response.json(),
                    headers=headers,
                    files=files
                )
            else:
                print(f"❌ Prédiction échouée: {response.status_code}")
                return self.log_test(
                    endpoint=endpoint,
                    method="POST",
                    status=response.status_code,
                    success=False,
                    request_data={"image_path": image_path},
                    response_data=response.json(),
                    headers=headers,
                    files=files
                )
                
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return self.log_test(
                endpoint=endpoint,
                method="POST",
                status=None,
                success=False,
                request_data={"image_path": image_path},
                error=e,
                headers=headers
            )

    def test_predict_batch(self, image_paths):
        """Test de l'endpoint predict_batch"""
        endpoint = "/predict_batch"
        try:
            print(f"\n📚 Test de {endpoint}")
            files = [('files', open(path, 'rb')) for path in image_paths]
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                files=files,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Prédiction batch réussie")
                return self.log_test(endpoint, "POST", response.status_code, True, 
                                   {"images": image_paths}, response.json(), headers=headers, files=files)
            else:
                print(f"❌ Prédiction batch échouée: {response.status_code}")
                return self.log_test(endpoint, "POST", response.status_code, False, 
                                   {"images": image_paths}, response.json(), headers=headers, files=files)
                
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return self.log_test(endpoint, "POST", None, False, 
                               {"images": image_paths}, None, e, headers=headers, files=files)

    def test_predict_zip(self, zip_path):
        """Test de l'endpoint predict_zip"""
        endpoint = "/predict_zip"
        try:
            print(f"\n📦 Test de {endpoint}")
            with open(zip_path, 'rb') as zip_file:
                files = {'zip_file': zip_file}
                headers = {'Authorization': f'Bearer {self.token}'}
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    files=files,
                    headers=headers
                )
            
            if response.status_code == 200:
                print("✅ Prédiction ZIP réussie")
                return self.log_test(endpoint, "POST", response.status_code, True, 
                                   {"zip_file": zip_path}, response.json(), headers=headers, files=files)
            else:
                print(f"❌ Prédiction ZIP échouée: {response.status_code}")
                return self.log_test(endpoint, "POST", response.status_code, False, 
                                   {"zip_file": zip_path}, response.json(), headers=headers, files=files)
                
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return self.log_test(endpoint, "POST", None, False, 
                               {"zip_file": zip_path}, None, e, headers=headers, files=files)

    def save_results(self):
        """Sauvegarde les résultats des tests"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "api-test-endpoint"
        os.makedirs(results_dir, exist_ok=True)
        
        # Sauvegarder en JSON
        json_path = os.path.join(results_dir, f"test_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.test_results, f, indent=4)
        
        # Sauvegarder en MD avec le nouveau format de nom
        md_path = os.path.join(results_dir, f"test-lancement_{timestamp}.md")
        
        with open(md_path, 'w') as f:
            f.write("# Résultats des Tests API\n\n")
            f.write(f"## Test du {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for result in self.test_results:
                f.write(f"### {result['method']} {result['endpoint']}\n")
                f.write(f"Status: {result['response']['status_code']}\n")
                f.write(f"Succès: {'✅' if result['success'] else '❌'}\n\n")
                
                f.write("#### Détails de la Requête\n")
                f.write("```json\n")
                f.write(json.dumps({
                    "url": result['request']['url'],
                    "method": result['method'],
                    "headers": result['request']['headers'],
                    "data": result['request']['data'],
                    "files": result['request']['files']
                }, indent=2))
                f.write("\n```\n\n")
                
                f.write("#### Réponse\n")
                f.write("```json\n")
                f.write(json.dumps(result['response']['data'], indent=2))
                f.write("\n```\n\n")
                
                if result['error']:
                    f.write("#### Erreur\n")
                    f.write(f"```\n{result['error']}\n```\n\n")
                
                f.write("---\n\n")
        
        print(f"\n📝 Résultats sauvegardés dans:")
        print(f"- {json_path}")
        print(f"- {md_path}")

def run_tests():
    """Exécute tous les tests"""
    tester = APITester()
    
    # Définir les chemins des fichiers de test avec le chemin absolu dans Docker
    test_dir = "/app/api-test-endpoint"  # Chemin absolu dans le conteneur
    test_files = {
        "image": os.path.join(test_dir, "images-test.jpg"),
        "zip": os.path.join(test_dir, "zip-test.zip"),
        "batch_images": [
            os.path.join(test_dir, "multi-image-test.jpg"),
            os.path.join(test_dir, "multi-image-test-2.jpg")
        ]
    }
    
    # Vérifier que les fichiers existent
    if not os.path.exists(test_files["image"]):
        print(f"❌ Erreur: Image de test non trouvée: {test_files['image']}")
        return
    
    if not os.path.exists(test_files["zip"]):
        print(f"❌ Erreur: ZIP de test non trouvé: {test_files['zip']}")
        return
        
    for img_path in test_files["batch_images"]:
        if not os.path.exists(img_path):
            print(f"❌ Erreur: Image batch non trouvée: {img_path}")
            return
    
    print("✅ Tous les fichiers de test sont accessibles")
    
    # Test de login
    tester.login("admin", "prodPassword")
    
    # Test de predict avec une image
    tester.test_predict(test_files["image"])
    
    # Test de predict_batch avec les deux images spécifiques
    tester.test_predict_batch(test_files["batch_images"])
    
    # Test de predict_zip avec le fichier ZIP
    tester.test_predict_zip(test_files["zip"])
    
    # Sauvegarder les résultats
    tester.save_results()

if __name__ == "__main__":
    run_tests() 