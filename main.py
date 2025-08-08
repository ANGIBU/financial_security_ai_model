# main.py
"""
개발용 메인 파일
"""

import os
import sys
import time
import pandas as pd
import torch
import argparse
import re
from pathlib import Path
from typing import Dict, List

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

def test_korean_quality(text: str) -> Dict:
    """한국어 품질 검사"""
    
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'[^\w]', '', text))
    korean_ratio = korean_chars / max(total_chars, 1)
    
    english_chars = len(re.findall(r'[A-Za-z]', text))
    english_ratio = english_chars / max(len(text), 1)
    
    return {
        "has_chinese": len(chinese_chars) > 0,
        "chinese_chars": chinese_chars,
        "korean_ratio": korean_ratio,
        "english_ratio": english_ratio,
        "is_good_quality": len(chinese_chars) == 0 and korean_ratio > 0.6 and english_ratio < 0.2
    }

def test_basic():
    """기본 시스템 체크"""
    print("기본 시스템 체크")
    print("="*50)
    
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
    else:
        print("GPU 없음")
    
    files = ["test.csv", "sample_submission.csv"]
    for f in files:
        if os.path.exists(f):
            print(f"파일 존재: {f}")
            df = pd.read_csv(f)
            print(f"  데이터 수: {len(df)}개")
        else:
            print(f"파일 없음: {f}")
    
    print("\n한국어 처리 모듈 테스트:")
    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        
        test_texts = [
            "이것은 软件입니다",
            "金融 거래의 安全성",
            "financial system 관리",
            "개인정보보호법에 따른 조치"
        ]
        
        for text in test_texts:
            cleaned = processor._clean_korean_text(text)
            quality = test_korean_quality(cleaned)
            print(f"  원본: {text}")
            print(f"  정리: {cleaned}")
            print(f"  품질: {'양호' if quality['is_good_quality'] else '개선필요'}")
            print()
            
    except Exception as e:
        print(f"  오류: {e}")
    
    print()

def test_korean_generation():
    """한국어 생성 테스트"""
    print("한국어 생성 품질 테스트")
    print("="*50)
    
    try:
        from model_handler import ModelHandler
        
        print("모델 로딩 중...")
        start = time.time()
        
        handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        load_time = time.time() - start
        print(f"모델 로딩 시간: {load_time:.1f}초")
        
        test_prompts = [
            {
                "name": "객관식 테스트",
                "prompt": """### 지시사항
당신은 한국의 금융보안 전문가입니다.
반드시 한국어로만 답변하고, 한자나 영어는 절대 사용하지 마세요.

### 문제
개인정보보호법상 개인정보의 정의로 가장 적절한 것은?
1. 모든 정보
2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보
3. 기업 정보
4. 공개된 정보
5. 암호화된 정보

반드시 한국어로만 간단히 분석한 후 정답 번호만 제시하세요.

정답 번호:""",
                "type": "multiple_choice"
            },
            {
                "name": "주관식 테스트",
                "prompt": """### 지시사항
당신은 한국의 금융보안 전문가입니다.
반드시 한국어로만 답변하세요. 한자, 영어, 기타 외국어는 절대 사용하지 마세요.

### 질문
개인정보보호 관리체계 구축 방안을 설명하세요.

### 중요 사항
- 전문적이고 정확한 한국어만 사용
- 한자나 영어 단어 사용 금지
- 법령명도 한국어로 표기

답변:""",
                "type": "subjective"
            }
        ]
        
        print("\n한국어 생성 테스트...")
        for test in test_prompts:
            print(f"\n{test['name']}")
            print("-" * 30)
            
            start_time = time.time()
            result = handler.generate_response(
                test["prompt"], 
                test["type"],
                max_attempts=1
            )
            elapsed = time.time() - start_time
            
            print(f"응답 시간: {elapsed:.2f}초")
            print(f"원본 응답: {result.response}")
            
            quality = test_korean_quality(result.response)
            print(f"\n품질 분석:")
            print(f"  한자 포함: {'예' if quality['has_chinese'] else '아니오'}")
            if quality['chinese_chars']:
                print(f"  한자 문자: {quality['chinese_chars']}")
            print(f"  한국어 비율: {quality['korean_ratio']:.2%}")
            print(f"  영어 비율: {quality['english_ratio']:.2%}")
            print(f"  품질 평가: {'우수' if quality['is_good_quality'] else '개선필요'}")
        
        handler.cleanup()
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

def test_small_inference_korean(sample_size: int = 5, with_learning: bool = False):
    """소규모 추론 테스트"""
    mode_text = "학습 포함" if with_learning else "기본"
    print(f"소규모 추론 테스트 ({sample_size}개, {mode_text} 모드)")
    print("="*50)
    
    if not os.path.exists("test.csv"):
        print("오류: test.csv 파일이 없습니다.")
        return
    
    from inference import FinancialAIInference
    
    test_df = pd.read_csv("test.csv")
    
    if len(test_df) < sample_size:
        sample_size = len(test_df)
        print(f"데이터 부족. {sample_size}개만 테스트합니다.")
    
    sample_df = test_df.head(sample_size).copy()
    sample_df.to_csv("test_sample.csv", index=False)
    
    engine = None
    try:
        start = time.time()
        print(f"\n모델 로딩 중...")
        engine = FinancialAIInference(enable_learning=with_learning)
        
        answers = []
        korean_quality_results = []
        
        for idx, row in sample_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            print(f"\n문제 {idx+1}/{sample_size}:")
            print(f"ID: {question_id}")
            print(f"질문: {question[:100]}")
            
            answer = engine.process_question(question, question_id, idx)
            answers.append(answer)
            
            quality = test_korean_quality(answer)
            korean_quality_results.append(quality)
            
            if len(answer) > 100:
                print(f"답변: {answer[:100]}")
            else:
                print(f"답변: {answer}")
            
            if quality['has_chinese']:
                print(f"한자 포함: {quality['chinese_chars']}")
            if quality['is_good_quality']:
                print("한국어 품질 우수")
            else:
                print("한국어 품질 개선 필요")
        
        elapsed = time.time() - start
        
        mc_answers = [a for a in answers if a.isdigit()]
        subj_answers = [a for a in answers if not a.isdigit()]
        
        total_with_chinese = sum(1 for q in korean_quality_results if q['has_chinese'])
        total_good_quality = sum(1 for q in korean_quality_results if q['is_good_quality'])
        avg_korean_ratio = sum(q['korean_ratio'] for q in korean_quality_results) / len(korean_quality_results)
        
        print("\n" + "="*50)
        print("테스트 결과")
        print("="*50)
        print(f"처리 시간: {elapsed:.1f}초")
        print(f"평균 시간: {elapsed/sample_size:.1f}초/문항")
        
        print(f"\n한국어 품질 리포트:")
        print(f"  한자 포함 답변: {total_with_chinese}/{sample_size}개")
        print(f"  품질 우수 답변: {total_good_quality}/{sample_size}개")
        print(f"  평균 한국어 비율: {avg_korean_ratio:.2%}")
        print(f"  전체 품질 점수: {(total_good_quality/sample_size)*100:.1f}점")
        
        if total_with_chinese == 0:
            print("  한자 혼재 문제 해결됨")
        else:
            print("  일부 답변에 한자 포함")
        
        if with_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {engine.stats.get('learned', 0)}개")
            print(f"  한국어 수정: {engine.stats.get('korean_fixes', 0)}회")
            if hasattr(engine, 'learning_system'):
                print(f"  정확도: {engine.learning_system.get_current_accuracy():.2%}")
        
        if mc_answers:
            print(f"\n객관식 답변 ({len(mc_answers)}개): {mc_answers}")
            distribution = {}
            for ans in mc_answers:
                distribution[ans] = distribution.get(ans, 0) + 1
            print("답변 분포:", distribution)
            
            if len(distribution) >= min(4, len(mc_answers)):
                print("다양한 답변 생성됨")
            else:
                print("답변 다양성 부족")
        
        if subj_answers:
            print(f"\n주관식 답변 ({len(subj_answers)}개)")
            for i, ans in enumerate(subj_answers[:3]):
                print(f"  {i+1}. {ans[:50]}")
                quality = korean_quality_results[len(mc_answers) + i] if len(mc_answers) + i < len(korean_quality_results) else None
                if quality and quality['has_chinese']:
                    print(f"     한자 포함: {quality['chinese_chars'][:3]}")
        
        total_questions = len(test_df)
        estimated_time = (elapsed / sample_size) * total_questions / 60
        print(f"\n예상 전체 시간: {estimated_time:.1f}분")
        
        if estimated_time > 270:
            print("시간 제한 초과 예상")
        else:
            print("시간 제한 내 완료 가능")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()
        
        if os.path.exists("test_sample.csv"):
            os.remove("test_sample.csv")

def test_patterns():
    """패턴 테스트"""
    print("패턴 학습 테스트")
    print("="*50)
    
    from pattern_learner import AnswerPatternLearner
    from data_processor import DataProcessor
    
    test_questions = [
        "개인정보란 살아 있는 개인에 관한 정보로서 다음 중 개인정보의 정의로 가장 적절한 것은?\n1. 모든 정보\n2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보\n3. 기업 정보\n4. 공개된 정보\n5. 암호화된 정보",
        "전자금융거래란 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래를 말한다. 다음 중 전자금융거래의 정의로 옳은 것은?\n1. 모든 거래\n2. 전자적 장치를 통한 금융상품 및 서비스 거래\n3. 인터넷만\n4. ATM만\n5. 현금거래",
        "다음 중 개인정보 유출 시 조치사항으로 적절하지 않은 것은?\n1. 즉시 통지\n2. 1주일 후 통지\n3. 개인정보보호위원회 신고\n4. 피해 최소화 조치\n5. 재발 방지 대책",
    ]
    
    learner = AnswerPatternLearner()
    processor = DataProcessor()
    
    print("한국어 처리 테스트:")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n테스트 {i}:")
        print(f"질문: {question[:80]}")
        
        cleaned = processor._clean_korean_text(question)
        quality = test_korean_quality(cleaned)
        print(f"한국어 품질: {'양호' if quality['is_good_quality'] else '개선필요'}")
        
        structure = processor.analyze_question_structure(question)
        pattern = learner.analyze_question_pattern(question)
        
        if pattern:
            print(f"매칭 패턴: {pattern['rule']}")
            print(f"신뢰도: {pattern['base_confidence']:.2f}")
            
            answer, confidence = learner.predict_answer(question, structure)
            print(f"예측 답: {answer}번 (신뢰도: {confidence:.2f})")
        else:
            print("패턴 매칭 없음")
            answer, confidence = learner.predict_answer(question, structure)
            print(f"통계적 예측: {answer}번 (신뢰도: {confidence:.2f})")

def test_manual_correction():
    """수동 교정 테스트"""
    print("수동 교정 시스템 테스트")
    print("="*50)
    
    from manual_correction import ManualCorrectionSystem
    
    correction_system = ManualCorrectionSystem()
    
    test_corrections = [
        {
            "question": "개인정보의 정의는?",
            "predicted": "1",
            "correct": "2",
            "reason": "법령 정의 확인"
        },
        {
            "question": "전자금융거래의 범위는?", 
            "predicted": "전자거래는 모든 거래입니다",
            "correct": "전자금융거래는 전자적 장치를 통한 금융상품 및 서비스 제공과 이용 거래입니다",
            "reason": "정확한 법령 정의 적용"
        }
    ]
    
    print("테스트 교정 데이터 추가:")
    for i, correction in enumerate(test_corrections, 1):
        correction_system.add_correction(
            question=correction["question"],
            predicted=correction["predicted"],
            correct=correction["correct"],
            reason=correction["reason"]
        )
        print(f"  {i}. {correction['question'][:30]} -> {correction['correct'][:20]}")
    
    print(f"\n교정 적용 테스트:")
    test_questions = [
        "개인정보의 정의는?",
        "개인정보란 무엇인가?",
        "전자금융거래의 범위는?",
        "새로운 질문입니다"
    ]
    
    for question in test_questions:
        corrected, confidence = correction_system.apply_corrections(question, "기본답변")
        print(f"  질문: {question}")
        print(f"  교정 결과: {corrected} (신뢰도: {confidence:.2f})")
        print()
    
    stats = correction_system.get_correction_stats()
    print(f"교정 통계:")
    print(f"  총 교정: {stats['total']}개")
    print(f"  객관식: {stats['mc']}개")
    print(f"  주관식: {stats['subjective']}개")
    print(f"  패턴: {stats['patterns']}개")

def show_correction_guide():
    """수동 교정 사용법 가이드"""
    print("수동 교정 시스템 사용법 가이드")
    print("="*50)
    
    guide_text = """
### 1. 교정 데이터 준비

**CSV 파일 형식** (corrections.csv):
```
id,question,predicted,correct,reason
Q001,개인정보의 정의는?,1,2,법령 정의 확인
Q002,전자금융 범위,잘못된답변,올바른답변,정확한 법령 적용
```

**JSON 파일 형식** (corrections.json):
```json
[
    {
        "id": "Q001",
        "question": "개인정보의 정의는?",
        "predicted": "1",
        "correct": "2", 
        "reason": "법령 정의 확인"
    }
]
```

### 2. 교정 시스템 사용

**기본 사용법:**
```python
from manual_correction import ManualCorrectionSystem

correction = ManualCorrectionSystem()

correction.add_correction(
    question="문제 내용",
    predicted="예측 답변", 
    correct="올바른 답변",
    reason="교정 이유"
)

corrected, confidence = correction.apply_corrections(question, predicted)
```

### 3. 대화형 교정

추론 실행 시 다음 옵션 활성화:
```python
engine.execute_inference(
    test_file="test.csv",
    submission_file="sample_submission.csv", 
    enable_manual_correction=True
)
```

### 4. 교정 효과

- 패턴 학습: 비슷한 문제에 자동 적용
- 신뢰도 증가: 교정된 답변의 신뢰도 향상  
- 학습 데이터: 향후 모델 학습에 활용
- 일관성: 동일한 패턴의 문제에 일관된 답변

### 5. 주의사항

- 교정 이유를 명확히 기록
- 법령이나 규정 근거 제시
- 과도한 교정은 오히려 성능 저하 가능
- 정기적인 교정 효과 검증 필요

### 6. 파일 위치

- 저장: `./corrections.csv`
- 로드: 시스템 시작 시 자동 로드
- 백업: 정기적으로 백업 권장
"""
    
    print(guide_text)

def test_speed():
    """속도 테스트"""
    print("속도 테스트")
    print("="*50)
    
    from model_handler import ModelHandler
    
    print("모델 로딩 중...")
    start = time.time()
    
    try:
        handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        load_time = time.time() - start
        print(f"모델 로딩 시간: {load_time:.1f}초")
        
        test_prompt = """### 지시사항
당신은 한국의 금융보안 전문가입니다.
반드시 한국어로만 답변하고, 한자나 영어는 절대 사용하지 마세요.

### 문제
개인정보보호법상 개인정보의 정의로 가장 적절한 것은?
1. 모든 정보
2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보
3. 기업 정보
4. 공개된 정보
5. 암호화된 정보

반드시 한국어로만 간단히 분석한 후 정답 번호만 제시하세요.

정답 번호:"""
        
        print("\n추론 테스트...")
        
        _ = handler.generate_response(test_prompt, "multiple_choice", max_attempts=1)
        
        times = []
        korean_quality_scores = []
        
        for i in range(3):
            start = time.time()
            result = handler.generate_response(test_prompt, "multiple_choice", max_attempts=1)
            elapsed = time.time() - start
            times.append(elapsed)
            
            quality = test_korean_quality(result.response)
            korean_quality_scores.append(quality['is_good_quality'])
            
            print(f"시도 {i+1}: {elapsed:.2f}초")
            print(f"  답변: {result.response[:50]}")
            print(f"  한국어 품질: {'양호' if quality['is_good_quality'] else '개선필요'}")
            if quality['has_chinese']:
                print(f"  한자 발견: {quality['chinese_chars'][:3]}")
        
        avg_time = sum(times) / len(times)
        korean_success_rate = sum(korean_quality_scores) / len(korean_quality_scores)
        
        print(f"\n성능 결과")
        print("="*30)
        print(f"평균 추론 시간: {avg_time:.2f}초")
        print(f"한국어 품질 성공률: {korean_success_rate:.2%}")
        print(f"예상 전체 시간 (515문항): {avg_time * 515 / 60:.1f}분")
        
        if korean_success_rate >= 0.9:
            print("한국어 품질 우수")
        else:
            print("한국어 품질 개선 필요")
        
        handler.cleanup()
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

def show_menu():
    """메뉴 표시"""
    print("실행 옵션:")
    print("1. 기본 시스템 체크")
    print("2. 한국어 생성 품질 테스트")
    print("3. 소규모 추론 테스트 (5문항)")
    print("4. 중규모 추론 테스트 (10문항)")
    print("5. 학습 포함 추론 테스트 (5문항)")
    print("6. 패턴 학습 테스트")
    print("7. 속도 성능 테스트")
    print("8. 수동 교정 테스트")
    print("9. 수동 교정 사용법 가이드")
    print("10. 전체 추론 실행 (515문항)")
    print("0. 종료")
    print("-"*50)

def interactive_mode():
    """대화형 모드"""
    show_menu()
    
    try:
        choice = input("\n선택하세요 (0-10): ").strip()
        
        if choice == "0":
            print("종료합니다.")
            return
        elif choice == "1":
            test_basic()
        elif choice == "2":
            test_korean_generation()
        elif choice == "3":
            test_small_inference_korean(5, with_learning=False)
        elif choice == "4":
            test_small_inference_korean(10, with_learning=False)
        elif choice == "5":
            test_small_inference_korean(5, with_learning=True)
        elif choice == "6":
            test_patterns()
        elif choice == "7":
            test_speed()
        elif choice == "8":
            test_manual_correction()
        elif choice == "9":
            show_correction_guide()
        elif choice == "10":
            confirm = input("전체 515문항을 실행합니다. 계속하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                os.system("python inference.py")
            else:
                print("취소되었습니다.")
        else:
            print("잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        print("\n\n종료합니다.")
    except Exception as e:
        print(f"오류: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="금융 AI 테스트")
    parser.add_argument("--test-type", choices=["basic", "korean", "small", "patterns", "speed", "correction", "interactive"], 
                       help="테스트 유형")
    parser.add_argument("--sample-size", type=int, default=5, 
                       help="샘플 크기 (small 테스트용)")
    parser.add_argument("--with-learning", action="store_true",
                       help="학습 기능 활성화")
    
    args = parser.parse_args()
    
    if not args.test_type:
        interactive_mode()
    elif args.test_type == "basic":
        test_basic()
    elif args.test_type == "korean":
        test_korean_generation()
    elif args.test_type == "small":
        test_small_inference_korean(args.sample_size, args.with_learning)
    elif args.test_type == "patterns":
        test_patterns()
    elif args.test_type == "speed":
        test_speed()
    elif args.test_type == "correction":
        test_manual_correction()
    elif args.test_type == "interactive":
        interactive_mode()

if __name__ == "__main__":
    main()5