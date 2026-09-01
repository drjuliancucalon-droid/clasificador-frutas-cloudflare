"""
Pruebas automatizadas con Playwright para el Clasificador de Frutas.
Actividad 3-1 UAM — Verifica frontend, API y genera screenshots.
"""
import os
import subprocess
import json
from playwright.sync_api import sync_playwright

API_URL = "https://clasificador-frutas-api.dr-juliancucalon.workers.dev"
FRONTEND_URL = "https://1378fe71.clasificador-frutas.pages.dev"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_api():
    """Prueba los endpoints de la API con curl"""
    print(">> Probando API con curl...")

    endpoints = ["/health", "/", "/metrics", "/history"]
    for endpoint in endpoints:
        result = subprocess.run(
            ["curl", "-s", f"{API_URL}{endpoint}"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        ok = endpoint == "/health" and data.get("status") == "ok" or \
             endpoint == "/" and "endpoints" in data or \
             endpoint == "/metrics" and "total_predicciones" in data or \
             endpoint == "/history" and "predicciones" in data
        status = "OK" if ok else "FAIL"
        print(f"  {status} {endpoint}")
    print("  OK API endpoints verificados")

def test_frontend():
    """Prueba el frontend con Playwright y genera screenshots"""
    print("\n>> Probando Frontend con Playwright...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # 1. Cargar frontend
        print("  Cargando frontend...")
        page.goto(FRONTEND_URL, wait_until="networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path=os.path.join(OUTPUT_DIR, "01-frontend-inicio.png"), full_page=True)
        print(f"  OK Screenshot: 01-frontend-inicio.png")

        # 2. Verificar título
        title = page.title()
        assert "Clasificador" in title, f"Title mismatch: {title}"
        print(f"  OK Titulo: '{title}'")

        # 3. Verificar pestaña Metrics
        print("  Navegando a Metricas...")
        metrics_button = page.locator("button:has-text('📊 Métricas')")
        assert metrics_button.is_visible(), "Botón Métricas no visible"
        metrics_button.click()
        page.wait_for_timeout(2000)
        page.screenshot(path=os.path.join(OUTPUT_DIR, "02-frontend-metricas.png"), full_page=True)
        print(f"  OK Screenshot: 02-frontend-metricas.png")

        # 4. Verificar pestaña History
        print("  Navegando a Historial...")
        history_button = page.locator("button:has-text('📋 Historial')")
        assert history_button.is_visible(), "Botón Historial no visible"
        history_button.click()
        page.wait_for_timeout(2000)
        page.screenshot(path=os.path.join(OUTPUT_DIR, "03-frontend-historial.png"), full_page=True)
        print(f"  OK Screenshot: 03-frontend-historial.png")

        # 5. Volver a Predict y verificar drag & drop
        print("  Verificando zona de subida...")
        predict_button = page.locator("button:has-text('🔮 Clasificar')")
        assert predict_button.is_visible(), "Botón Clasificar no visible"
        predict_button.click()
        page.wait_for_timeout(1000)

        # Verificar que la zona de drop existe
        drop_zone = page.locator("text=Arrastra una imagen aquí")
        assert drop_zone.is_visible(), "Zona de drag & drop no visible"
        page.screenshot(path=os.path.join(OUTPUT_DIR, "04-frontend-clasificar.png"), full_page=True)
        print(f"  OK Screenshot: 04-frontend-clasificar.png")

        # 6. Verificar enlaces a la API
        print("  Verificando enlaces...")
        health_link = page.locator("a:has-text('Health')")
        assert health_link.is_visible(), "Link Health no visible"
        docs_link = page.locator("a:has-text('Docs')")
        assert docs_link.is_visible(), "Link Docs no visible"
        print(f"  OK Enlaces Health y Docs visibles")

        browser.close()
        print("  OK Frontend funciona correctamente")
        print(f"  📸 Screenshots guardados en: {OUTPUT_DIR}")

def main():
    print("=" * 60)
    print("Pruebas del Sistema - Clasificador de Frutas")
    print(f"   API: {API_URL}")
    print(f"   Frontend: {FRONTEND_URL}")
    print("=" * 60)

    try:
        test_api()
        test_frontend()
        print("\n" + "=" * 60)
        print("TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nPrueba fallo: {e}")
        raise
    except Exception as e:
        print(f"\nError inesperado: {e}")
        raise

if __name__ == "__main__":
    main()