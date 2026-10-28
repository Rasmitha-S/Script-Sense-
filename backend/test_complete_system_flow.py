import io
import json
from fastapi.testclient import TestClient

from main import app
from database import SessionLocal
import models

client = TestClient(app)

def test_full_system_flow():
    print("=" * 60)
    print("STARTING FULL END-TO-END SCRIPTSENSE SYSTEM VERIFICATION")
    print("=" * 60)

    # 1. Health check
    res = client.get("/")
    assert res.status_code == 200, f"Root failed: {res.text}"
    print("[1] Health check OK:", res.json())

    # 2. Login Teacher
    login_res = client.post("/api/login", json={"username": "teacher1", "password": "secret123", "role": "teacher"})
    assert login_res.status_code == 200, f"Teacher login failed: {login_res.text}"
    token_data = login_res.json()
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[2] Teacher login OK, user:", token_data["username"], "role:", token_data["role"])

    # 3. Get Current User
    me_res = client.get("/api/me", headers=headers)
    assert me_res.status_code == 200, f"Me endpoint failed: {me_res.text}"
    print("[3] /api/me OK:", me_res.json()["username"])

    # 4. Create or Get Student
    import time
    unique_roll = f"AJ-{int(time.time())}"
    student_res = client.post("/api/students", json={"name": "Alice Johnson", "roll_number": unique_roll}, headers=headers)
    assert student_res.status_code in [200, 201], f"Create student failed: {student_res.text}"
    student_data = student_res.json()
    student_id = student_data["id"]
    print("[4] Student created OK:", student_data["name"], "ID:", student_id, "Roll:", unique_roll)

    # 5. Create Test with Questions & Rubrics
    test_payload = {
        "test_name": "Physics Final Exam 2026",
        "subject": "Physics",
        "max_marks": 15.0,
        "questions": [
            {
                "q_num": 1,
                "question": "State Newton's second law of motion and give its equation.",
                "model_answer": "Newton's second law states that the rate of change of momentum is directly proportional to the applied force. Equation: F = m * a.",
                "max_marks": 10.0,
                "rubric": [
                    {"id": "r1", "criterion": "Statement of momentum change", "max_marks": 5.0, "keywords": ["momentum", "force", "proportional"]},
                    {"id": "r2", "criterion": "Mathematical equation F=ma", "max_marks": 5.0, "keywords": ["F = m * a", "mass", "acceleration"]}
                ]
            },
            {
                "q_num": 2,
                "question": "What is the SI unit of force?",
                "model_answer": "The SI unit of force is the Newton (N).",
                "max_marks": 5.0,
                "rubric": [
                    {"id": "r3", "criterion": "Identifies Newton correctly", "max_marks": 5.0, "keywords": ["newton", "N"]}
                ]
            }
        ],
        "student_ids": [student_id]
    }
    create_test_res = client.post("/api/tests", json=test_payload, headers=headers)
    assert create_test_res.status_code == 201, f"Create test failed: {create_test_res.text}"
    test_data = create_test_res.json()
    test_id = test_data["id"]
    model_answer_id = test_data["model_answer_id"]
    print("[5] Test created OK:", test_data["test_name"], "ID:", test_id, "Max Marks:", test_data["max_marks"])

    # 6. Verify Test in /api/tests and /api/tests/overview
    tests_res = client.get("/api/tests", headers=headers)
    assert tests_res.status_code == 200
    assert any(t["id"] == test_id for t in tests_res.json())
    print("[6a] /api/tests OK, found created test")

    overview_res = client.get("/api/tests/overview", headers=headers)
    assert overview_res.status_code == 200
    t_overview = next(t for t in overview_res.json() if t["id"] == test_id)
    assert t_overview["students_count"] >= 1
    print("[6b] /api/tests/overview OK: students count =", t_overview["students_count"])

    # 7. Upload Student Answer Sheet
    fake_img = io.BytesIO(b"Fake handwritten test sheet content")
    fake_img.name = "answer_sheet_alice.png"
    upload_res = client.post(
        "/api/upload",
        files={"file": ("answer_sheet_alice.png", fake_img, "image/png")},
        data={"student_id": str(student_id), "student_name": "Alice Johnson", "roll_number": unique_roll, "test_id": str(test_id)},
        headers=headers
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()
    sheet_id = upload_data["answer_sheet_id"]
    print("[7] Upload Answer Sheet OK, ID:", sheet_id, "Extracted:", upload_data["extracted_text"][:40])

    # 8. Edit OCR Transcript (Transliteration/Correction capability)
    edited_text = (
        "Q1: Newton's second law states that the rate of change of momentum is proportional to the applied force. "
        "The formula is F = m * a where F is force, m is mass and a is acceleration.\n\n"
        "Q2: The SI unit of force is the Newton."
    )
    transcript_res = client.put(f"/api/uploads/{sheet_id}/transcript", json={"extracted_text": edited_text}, headers=headers)
    assert transcript_res.status_code == 200
    print("[8] Update Transcript OK, edited text saved")

    # 9. Run AI Evaluation
    eval_res = client.post("/api/evaluate", json={"answer_sheet_id": sheet_id, "model_answer_id": model_answer_id}, headers=headers)
    assert eval_res.status_code == 200, f"Evaluation failed: {eval_res.text}"
    eval_data = eval_res.json()
    eval_id = eval_data["evaluation_id"]
    print("[9] Evaluation OK, suggested marks:", eval_data["suggested_marks"], "similarity:", eval_data["similarity"])

    # 10. Teacher Verification & Override Marks
    verify_res = client.put(
        f"/api/results/{eval_id}",
        json={"final_marks": 14.5, "teacher_feedback": "Excellent conceptual clarity and precise formula."},
        headers=headers
    )
    assert verify_res.status_code == 200, f"Verification failed: {verify_res.text}"
    print("[10] Verification OK, final marks:", verify_res.json()["final_marks"])

    # 11. Retrieve Result
    res_detail = client.get(f"/api/results/{eval_id}", headers=headers)
    assert res_detail.status_code == 200
    res_data = res_detail.json()
    assert res_data["final_marks"] == 14.5
    assert res_data["verified_by"] is not None
    print("[11] /api/results/{id} OK, verified by:", res_data["verified_by"])

    # 12. Batch Evaluate all sheets for Test
    batch_res = client.post(f"/api/tests/{test_id}/evaluate-all", headers=headers)
    assert batch_res.status_code == 200
    print("[12] /api/tests/{id}/evaluate-all OK, processed:", batch_res.json()["processed_count"])

    # 13. Student Portal Access
    portal_res = client.post("/api/student/portal-access", json={"roll_number_or_id": unique_roll})
    assert portal_res.status_code == 200, f"Portal lookup failed: {portal_res.text}"
    portal_data = portal_res.json()
    assert portal_data["student_name"] == "Alice Johnson"
    assert len(portal_data["results"]) >= 1
    print("[13] Student Portal lookup OK, student:", portal_data["student_name"], "results count:", len(portal_data["results"]))

    # 14. Explain Marks AI Chatbot
    explain_res = client.post(
        "/api/explain-marks",
        json={
            "student_answer": edited_text,
            "model_answer": "F = m * a",
            "similarity": 0.95,
            "marks_obtained": 14.5,
            "max_marks": 15.0,
            "explanation": "High alignment",
            "user_question": "Why did I lose 0.5 marks?"
        }
    )
    assert explain_res.status_code == 200
    safe_reply = explain_res.json()["reply"][:60].encode("ascii", errors="replace").decode("ascii")
    print("[14] /api/explain-marks AI Chatbot OK, reply:", safe_reply)

    # 15. Admin Check
    admin_login_res = client.post("/api/login", json={"username": "admin", "password": "admin123", "role": "admin"})
    assert admin_login_res.status_code == 200, f"Admin login failed: {admin_login_res.text}"
    admin_headers = {"Authorization": f"Bearer {admin_login_res.json()['access_token']}"}
    stats_res = client.get("/api/admin/stats", headers=admin_headers)
    assert stats_res.status_code == 200
    print("[15] Admin /api/admin/stats OK:", stats_res.json())

    print("\n" + "=" * 60)
    print("ALL 15 END-TO-END FUNCTIONALITY AND SYSTEM CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_system_flow()
