import os
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session

from database import get_db
import models
import schemas
from routers.auth import get_current_user
from services.file_service import validate_and_save_file
from services.ocr_service import extract_text_from_file
from services.evaluation_service import (
    evaluate_answers,
    evaluate_multi_question_exam,
)

router = APIRouter(prefix="/api/tests", tags=["Tests"])


@router.post("", response_model=schemas.TestResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=schemas.TestResponse, status_code=status.HTTP_201_CREATED)
def create_test(
    request: schemas.TestCreateRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Creates a new test, configures its model answer / questions / rubrics, and assigns students.
    """
    test_name_clean = request.test_name.strip() if request.test_name else ""
    if not test_name_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Test name cannot be empty."
        )

    questions_json_str = None
    rubric_json_str = None
    questions_count = 1
    total_max_marks = float(request.max_marks or 10.0)
    primary_question = ""
    primary_answer = ""

    if request.questions and len(request.questions) > 0:
        questions_dict = [q.model_dump() for q in request.questions]
        questions_json_str = json.dumps(questions_dict)
        calc_marks = sum(float(q.max_marks or 5.0) for q in request.questions)
        total_max_marks = calc_marks
        questions_count = len(request.questions)
        primary_question = f"{test_name_clean} — " + " | ".join(
            f"Q{q.q_num or i+1}: {q.question[:50]}" for i, q in enumerate(request.questions)
        )
        primary_answer = "\n\n".join(
            f"[Q{q.q_num or i+1} Model Solution]:\n{q.model_answer}" for i, q in enumerate(request.questions)
        )
    else:
        primary_question = request.question.strip() if (request.question and request.question.strip()) else test_name_clean
        primary_answer = request.answer_text.strip() if (request.answer_text and request.answer_text.strip()) else ""
        if request.rubric:
            rubric_dict = [r.model_dump() for r in request.rubric]
            rubric_json_str = json.dumps(rubric_dict)

    # 1. Create Test instance
    new_test = models.Test(
        test_name=test_name_clean,
        teacher_id=current_user.id,
        subject=request.subject.strip() if request.subject else "General",
        max_marks=total_max_marks,
    )
    db.add(new_test)
    db.commit()
    db.refresh(new_test)

    # 2. Create ModelAnswer linked to test
    model_answer = models.ModelAnswer(
        test_id=new_test.id,
        title=test_name_clean,
        subject=request.subject.strip() if request.subject else "General",
        question=primary_question,
        answer_text=primary_answer,
        max_marks=total_max_marks,
        questions_json=questions_json_str,
        rubric_json=rubric_json_str,
        file_path=request.file_path,
    )
    db.add(model_answer)
    db.commit()
    db.refresh(model_answer)

    # 3. Assign existing students belonging to this teacher
    assigned_students: List[models.Student] = []
    if request.student_ids:
        for s_id in request.student_ids:
            st = db.query(models.Student).filter(
                models.Student.id == s_id,
                models.Student.teacher_id == current_user.id
            ).first()
            if st and st not in assigned_students:
                new_test.students.append(st)
                assigned_students.append(st)

    # 4. Create and assign new students if requested
    if request.new_students:
        for new_st in request.new_students:
            name_clean = (new_st.name or "").strip()
            if not name_clean:
                continue
            roll_clean = new_st.roll_number.strip() if (new_st.roll_number and new_st.roll_number.strip()) else None

            st_obj = None
            if roll_clean:
                st_obj = db.query(models.Student).filter(
                    models.Student.teacher_id == current_user.id,
                    models.Student.roll_number == roll_clean
                ).first()

            if not st_obj:
                st_obj = models.Student(
                    teacher_id=current_user.id,
                    name=name_clean,
                    roll_number=roll_clean
                )
                db.add(st_obj)
                db.commit()
                db.refresh(st_obj)

            if st_obj not in assigned_students:
                new_test.students.append(st_obj)
                assigned_students.append(st_obj)

    db.commit()
    db.refresh(new_test)

    parsed_questions = None
    if questions_json_str:
        try:
            parsed_questions = json.loads(questions_json_str)
        except Exception:
            pass

    return schemas.TestResponse(
        id=new_test.id,
        test_name=new_test.test_name,
        teacher_id=new_test.teacher_id,
        subject=new_test.subject,
        max_marks=new_test.max_marks,
        created_at=new_test.created_at,
        questions_count=questions_count,
        students_count=len(assigned_students),
        model_answer_id=model_answer.id,
        students=[
            schemas.StudentResponse(
                id=s.id,
                teacher_id=s.teacher_id,
                name=s.name,
                roll_number=s.roll_number,
                created_at=s.created_at
            ) for s in assigned_students
        ],
        questions=parsed_questions
    )


@router.get("", response_model=List[schemas.TestResponse])
@router.get("/", response_model=List[schemas.TestResponse])
def get_tests(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Returns list of tests belonging to the current teacher.
    """
    query = db.query(models.Test)
    if current_user.role != "admin":
        query = query.filter(models.Test.teacher_id == current_user.id)
    tests = query.order_by(models.Test.id.desc()).all()

    result = []
    for t in tests:
        ma = db.query(models.ModelAnswer).filter(models.ModelAnswer.test_id == t.id).first()
        parsed_questions = None
        q_count = 1
        if ma and ma.questions_json:
            try:
                parsed_questions = json.loads(ma.questions_json)
                q_count = len(parsed_questions) if isinstance(parsed_questions, list) else 1
            except Exception:
                pass

        test_students = list(t.students)
        result.append(schemas.TestResponse(
            id=t.id,
            test_name=t.test_name,
            teacher_id=t.teacher_id,
            subject=t.subject,
            max_marks=t.max_marks,
            created_at=t.created_at,
            questions_count=q_count,
            students_count=len(test_students),
            model_answer_id=ma.id if ma else None,
            students=[
                schemas.StudentResponse(
                    id=s.id,
                    teacher_id=s.teacher_id,
                    name=s.name,
                    roll_number=s.roll_number,
                    created_at=s.created_at
                ) for s in test_students
            ],
            questions=parsed_questions
        ))
    return result


@router.get("/overview", response_model=List[schemas.TestOverviewResponse])
def get_tests_overview(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Returns comprehensive batch status overview for all tests created by the teacher.
    """
    query = db.query(models.Test)
    if current_user.role != "admin":
        query = query.filter(models.Test.teacher_id == current_user.id)
    tests = query.order_by(models.Test.id.desc()).all()

    overview_list = []
    for t in tests:
        ma = db.query(models.ModelAnswer).filter(models.ModelAnswer.test_id == t.id).first()
        q_count = 1
        if ma and ma.questions_json:
            try:
                parsed = json.loads(ma.questions_json)
                q_count = len(parsed) if isinstance(parsed, list) else 1
            except Exception:
                pass

        test_students = list(t.students)
        test_sheets = db.query(models.AnswerSheet).filter(models.AnswerSheet.test_id == t.id).all()
        sheet_by_student = {s.student_id: s for s in test_sheets if s.student_id is not None}

        uploaded_count = 0
        evaluated_count = 0
        verified_count = 0
        student_statuses = []

        for st in test_students:
            sheet = sheet_by_student.get(st.id)
            status_val = "Pending Upload"
            sheet_id = None
            eval_id = None
            sim_val = None
            sug_marks = None
            fin_marks = None
            ver_by = None
            ver_at = None
            up_at = None

            if sheet:
                sheet_id = sheet.id
                up_at = sheet.uploaded_at.isoformat() if sheet.uploaded_at else None
                evals = db.query(models.Evaluation).filter(
                    models.Evaluation.answer_sheet_id == sheet.id
                ).order_by(models.Evaluation.id.desc()).all()

                if evals:
                    latest_eval = evals[0]
                    eval_id = latest_eval.id
                    sim_val = latest_eval.similarity
                    sug_marks = latest_eval.suggested_marks

                    final_res = db.query(models.FinalResult).filter(
                        models.FinalResult.evaluation_id == latest_eval.id
                    ).first()

                    if final_res and final_res.final_marks is not None:
                        status_val = "Verified"
                        fin_marks = final_res.final_marks
                        ver_by = final_res.verified_by
                        ver_at = final_res.verified_at.isoformat() if final_res.verified_at else None
                        verified_count += 1
                    else:
                        status_val = "Evaluated"
                        evaluated_count += 1
                else:
                    status_val = "Uploaded"
                    uploaded_count += 1

            student_statuses.append(schemas.TestStudentStatus(
                student_id=st.id,
                student_name=st.name,
                roll_number=st.roll_number,
                answer_sheet_id=sheet_id,
                evaluation_id=eval_id,
                status=status_val,
                similarity=sim_val,
                suggested_marks=sug_marks,
                final_marks=fin_marks,
                max_marks=t.max_marks,
                verified_by=ver_by,
                verified_at=ver_at,
                uploaded_at=up_at
            ))

        overview_list.append(schemas.TestOverviewResponse(
            id=t.id,
            test_name=t.test_name,
            teacher_id=t.teacher_id,
            subject=t.subject,
            max_marks=t.max_marks,
            created_at=t.created_at,
            model_answer_id=ma.id if ma else None,
            questions_count=q_count,
            students_count=len(test_students),
            uploaded_count=uploaded_count,
            evaluated_count=evaluated_count,
            verified_count=verified_count,
            students=student_statuses
        ))

    return overview_list


@router.get("/{id}", response_model=schemas.TestResponse)
def get_test_details(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Returns single test details.
    """
    test = db.query(models.Test).filter(models.Test.id == id).first()
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test with ID {id} not found."
        )

    if current_user.role != "admin" and test.teacher_id is not None and test.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to view tests belonging to another teacher."
        )

    ma = db.query(models.ModelAnswer).filter(models.ModelAnswer.test_id == test.id).first()
    parsed_questions = None
    q_count = 1
    if ma and ma.questions_json:
        try:
            parsed_questions = json.loads(ma.questions_json)
            q_count = len(parsed_questions) if isinstance(parsed_questions, list) else 1
        except Exception:
            pass

    test_students = list(test.students)
    return schemas.TestResponse(
        id=test.id,
        test_name=test.test_name,
        teacher_id=test.teacher_id,
        subject=test.subject,
        max_marks=test.max_marks,
        created_at=test.created_at,
        questions_count=q_count,
        students_count=len(test_students),
        model_answer_id=ma.id if ma else None,
        students=[
            schemas.StudentResponse(
                id=s.id,
                teacher_id=s.teacher_id,
                name=s.name,
                roll_number=s.roll_number,
                created_at=s.created_at
            ) for s in test_students
        ],
        questions=parsed_questions
    )


@router.post("/{id}/students", response_model=schemas.TestResponse)
def assign_students_to_test(
    id: int,
    request: schemas.TestStudentAssignRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Assigns existing or new students to an existing test.
    """
    test = db.query(models.Test).filter(models.Test.id == id).first()
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test with ID {id} not found."
        )

    if current_user.role != "admin" and test.teacher_id is not None and test.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot modify tests belonging to another teacher."
        )

    if request.student_ids:
        for s_id in request.student_ids:
            st = db.query(models.Student).filter(
                models.Student.id == s_id,
                models.Student.teacher_id == current_user.id
            ).first()
            if st and st not in test.students:
                test.students.append(st)

    if request.new_students:
        for new_st in request.new_students:
            name_clean = (new_st.name or "").strip()
            if not name_clean:
                continue
            roll_clean = new_st.roll_number.strip() if (new_st.roll_number and new_st.roll_number.strip()) else None

            st_obj = None
            if roll_clean:
                st_obj = db.query(models.Student).filter(
                    models.Student.teacher_id == current_user.id,
                    models.Student.roll_number == roll_clean
                ).first()

            if not st_obj:
                st_obj = models.Student(
                    teacher_id=current_user.id,
                    name=name_clean,
                    roll_number=roll_clean
                )
                db.add(st_obj)
                db.commit()
                db.refresh(st_obj)

            if st_obj not in test.students:
                test.students.append(st_obj)

    db.commit()
    db.refresh(test)

    ma = db.query(models.ModelAnswer).filter(models.ModelAnswer.test_id == test.id).first()
    test_students = list(test.students)
    return schemas.TestResponse(
        id=test.id,
        test_name=test.test_name,
        teacher_id=test.teacher_id,
        subject=test.subject,
        max_marks=test.max_marks,
        created_at=test.created_at,
        questions_count=1,
        students_count=len(test_students),
        model_answer_id=ma.id if ma else None,
        students=[
            schemas.StudentResponse(
                id=s.id,
                teacher_id=s.teacher_id,
                name=s.name,
                roll_number=s.roll_number,
                created_at=s.created_at
            ) for s in test_students
        ]
    )


@router.post("/{id}/evaluate-all", response_model=schemas.BatchEvaluateResponse)
def evaluate_all_test_sheets(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Evaluates all uploaded answer sheets for this test against its model answer.
    """
    test = db.query(models.Test).filter(models.Test.id == id).first()
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test with ID {id} not found."
        )

    if current_user.role != "admin" and test.teacher_id is not None and test.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot evaluate tests belonging to another teacher."
        )

    model_answer = db.query(models.ModelAnswer).filter(models.ModelAnswer.test_id == id).first()
    if not model_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This test does not have a configured model answer."
        )

    sheets = db.query(models.AnswerSheet).filter(models.AnswerSheet.test_id == id).all()
    if not sheets:
        return schemas.BatchEvaluateResponse(
            processed_count=0,
            successful_evaluations=[],
            failed_ids=[]
        )

    questions_list = []
    if model_answer.questions_json:
        try:
            questions_list = json.loads(model_answer.questions_json)
        except Exception:
            pass

    rubric_list = None
    if model_answer.rubric_json:
        try:
            rubric_list = json.loads(model_answer.rubric_json)
        except Exception:
            pass

    successful = []
    failed_ids = []

    for sheet in sheets:
        try:
            student_text = sheet.extracted_text or ""
            similarity = 0.0
            suggested_marks = 0.0
            explanation = ""
            rubric_scores = []
            question_evaluations = []

            if questions_list and len(questions_list) > 0:
                sim, marks, expl, r_scores, q_evals = evaluate_multi_question_exam(
                    student_text, questions_list, model_answer.max_marks
                )
                similarity = sim
                suggested_marks = marks
                explanation = expl
                rubric_scores = r_scores
                question_evaluations = q_evals
            else:
                sim, marks, expl, r_scores = evaluate_answers(
                    student_text, model_answer.answer_text, model_answer.max_marks, rubric_list
                )
                similarity = sim
                suggested_marks = marks
                explanation = expl
                rubric_scores = r_scores

            existing_eval = db.query(models.Evaluation).filter(
                models.Evaluation.answer_sheet_id == sheet.id,
                models.Evaluation.model_answer_id == model_answer.id
            ).first()

            rubric_scores_json = json.dumps(rubric_scores) if rubric_scores else None
            question_evals_json = json.dumps(question_evaluations) if question_evaluations else None

            if existing_eval:
                existing_eval.similarity = similarity
                existing_eval.suggested_marks = suggested_marks
                existing_eval.explanation = explanation
                existing_eval.rubric_scores_json = rubric_scores_json
                existing_eval.question_evaluations_json = question_evals_json
                db.commit()
                db.refresh(existing_eval)
                eval_record = existing_eval
            else:
                eval_record = models.Evaluation(
                    answer_sheet_id=sheet.id,
                    model_answer_id=model_answer.id,
                    similarity=similarity,
                    suggested_marks=suggested_marks,
                    explanation=explanation,
                    rubric_scores_json=rubric_scores_json,
                    question_evaluations_json=question_evals_json,
                )
                db.add(eval_record)
                db.commit()
                db.refresh(eval_record)

            st_obj = db.query(models.Student).filter(models.Student.id == sheet.student_id).first() if sheet.student_id else None

            successful.append(schemas.EvaluateResponse(
                evaluation_id=eval_record.id,
                answer_sheet_id=sheet.id,
                model_answer_id=model_answer.id,
                test_id=test.id,
                test_name=test.test_name,
                student_id=sheet.student_id,
                student_name=st_obj.name if st_obj else sheet.student_name,
                roll_number=st_obj.roll_number if st_obj else None,
                title=test.test_name,
                similarity=similarity,
                suggested_marks=suggested_marks,
                max_marks=model_answer.max_marks,
                explanation=explanation,
                rubric_scores=[schemas.RubricScoreSchema(**r) for r in rubric_scores] if rubric_scores else None,
                question_evaluations=[schemas.QuestionEvaluationSchema(**q) for q in question_evaluations] if question_evaluations else None,
            ))
        except Exception:
            failed_ids.append(sheet.id)

    return schemas.BatchEvaluateResponse(
        processed_count=len(successful),
        successful_evaluations=successful,
        failed_ids=failed_ids
    )


@router.delete("/{id}", response_model=schemas.MessageResponse)
def delete_test(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Deletes a test and unbinds its associated answer sheets.
    """
    test = db.query(models.Test).filter(models.Test.id == id).first()
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test with ID {id} not found."
        )

    if current_user.role != "admin" and test.teacher_id is not None and test.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to delete tests belonging to another teacher."
        )

    test_name = test.test_name
    db.delete(test)
    db.commit()

    return schemas.MessageResponse(
        message=f"Test '{test_name}' (ID: {id}) deleted successfully.",
        success=True
    )


@router.post("/extract-model-answer")
async def extract_model_answer(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Uploads a model answer file (PDF/Image) and extracts reference text using OCR.
    """
    try:
        relative_path, abs_path = validate_and_save_file(file, prefix_id="model_extract")
        ext = os.path.splitext(file.filename or "")[1].lower().replace(".", "") or "txt"
        try:
            extracted_text = extract_text_from_file(abs_path)
        except Exception:
            extracted_text = ""

        return {
            "success": True,
            "status": "success",
            "filename": file.filename,
            "saved_filename": os.path.basename(abs_path),
            "text": extracted_text,
            "extracted_text": extracted_text,
            "file_type": ext,
            "file_path": relative_path
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract model answer text: {str(e)}"
        )
