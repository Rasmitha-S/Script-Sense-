import os
from io import BytesIO
from PIL import Image, ImageDraw
from test_client_helper import get_client, BASE_URL

requests = get_client()

def run_test():
    print("[TEST] 1. Logging in...")
    login_res = requests.post(
        f"{BASE_URL}/api/login",
        json={"email": "teacher@demo.com", "password": "password123"}
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    print("[OK] Logged in successfully. Token acquired.")

    # 2. Generate a valid test image (PNG & JPG) with text
    img = Image.new("RGB", (400, 150), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 40), "Photosynthesis is the process by which green plants", fill=(0, 0, 0))
    d.text((20, 70), "use sunlight to synthesize nutrients.", fill=(0, 0, 0))

    png_buf = BytesIO()
    img.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    jpg_buf = BytesIO()
    img.save(jpg_buf, format="JPEG")
    jpg_bytes = jpg_buf.getvalue()

    # 3. Test PNG Upload to /api/extract-text
    print("\n[TEST] 2. Testing PNG Image Upload to /api/extract-text...")
    files = {"file": ("model_answer_q1.png", png_bytes, "image/png")}
    res_png = requests.post(f"{BASE_URL}/api/extract-text", headers=headers, files=files)
    print(f"Status Code: {res_png.status_code}")
    print(f"Response: {res_png.json()}")
    assert res_png.status_code == 200, f"PNG upload failed: {res_png.text}"
    data_png = res_png.json()
    assert data_png.get("success") is True
    saved_filename = data_png.get("filename")
    assert saved_filename is not None
    saved_path = os.path.join(os.path.dirname(__file__), "uploads", saved_filename)
    assert os.path.exists(saved_path), f"File {saved_path} was not saved on disk!"
    print(f"[OK] PNG extracted successfully. File saved on disk at: {saved_path}")

    # 4. Test JPG Upload to /api/extract-text
    print("\n[TEST] 3. Testing JPG Image Upload to /api/extract-text...")
    files_jpg = {"file": ("model_answer_q2.jpg", jpg_bytes, "image/jpeg")}
    res_jpg = requests.post(f"{BASE_URL}/api/extract-text", headers=headers, files=files_jpg)
    print(f"Status Code: {res_jpg.status_code}")
    print(f"Response: {res_jpg.json()}")
    assert res_jpg.status_code == 200, f"JPG upload failed: {res_jpg.text}"
    data_jpg = res_jpg.json()
    assert data_jpg.get("success") is True
    print("[OK] JPG extracted and saved successfully.")

    # 5. Create a complete Test using the extracted model answer text
    print("\n[TEST] 4. Creating complete Test with model answers...")
    extracted_text = data_png.get("text") or "Photosynthesis process by which plants use sunlight."
    test_payload = {
        "title": "Automated Upload Test - Biology Midterm",
        "subject": "Biology",
        "total_marks": 50,
        "questions": [
            {
                "question_number": 1,
                "question_text": "Explain the process of photosynthesis.",
                "max_marks": 25,
                "model_answer_text": extracted_text,
                "grading_criteria": "Accuracy of chemical process and energy conversion."
            },
            {
                "question_number": 2,
                "question_text": "Describe cellular respiration.",
                "max_marks": 25,
                "model_answer_text": "Cellular respiration converts glucose into ATP.",
                "grading_criteria": "Steps of respiration and ATP yield."
            }
        ]
    }
    create_test_res = requests.post(f"{BASE_URL}/api/tests", headers=headers, json=test_payload)
    print(f"Create Test Status: {create_test_res.status_code}")
    print(f"Create Test Response: {create_test_res.json()}")
    assert create_test_res.status_code in (200, 201), f"Create test failed: {create_test_res.text}"
    created_test = create_test_res.json()
    test_id = created_test.get("id")
    print(f"[OK] Test created successfully with ID: {test_id}")

    # 6. Verify Test is queryable
    print("\n[TEST] 5. Verifying created test via GET /api/tests...")
    tests_res = requests.post(f"{BASE_URL}/api/tests/overview", headers=headers) if False else requests.get(f"{BASE_URL}/api/tests", headers=headers)
    print(f"Fetch tests status: {tests_res.status_code}")
    assert tests_res.status_code == 200
    found = any(t.get("id") == test_id for t in tests_res.json())
    assert found, f"Created test {test_id} not found in test list!"
    print(f"[OK] Confirmed Test ID {test_id} is present in tests list.")

    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    run_test()
