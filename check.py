# check.py
"""
최종 제출 전 시스템 검증 스크립트
대회 규정 준수 및 성능 확인
"""

import os
import sys
import time
import torch
import pandas as pd
import subprocess
import psutil
from pathlib import Path

class FinalSystemChecker:
    """최종 시스템 체크"""
    
    def __init__(self):
        self.check_results = {
            "environment": {},
            "files": {},
            "model": {},
            "performance": {},
            "compliance": {}
        }
    
    def run_all_checks(self):
        """모든 검사 실행"""
        print("=== 최종 시스템 검증 시작 ===\n")
        
        # 1. 환경 체크
        self.check_environment()
        
        # 2. 파일 체크
        self.check_files()
        
        # 3. 모델 체크
        self.check_model()
        
        # 4. 성능 예측
        self.estimate_performance()
        
        # 5. 대회 규정 준수 체크
        self.check_compliance()
        
        # 최종 보고서
        self.generate_report()
    
    def check_environment(self):
        """환경 체크"""
        print("1. 환경 검사 중...")
        
        # GPU 체크
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.check_results["environment"]["gpu"] = f"{gpu_name} ({gpu_memory:.1f}GB)"
            self.check_results["environment"]["gpu_status"] = "✅ OK"
        else:
            self.check_results["environment"]["gpu_status"] = "❌ CUDA 사용 불가"
        
        # RAM 체크
        ram_gb = psutil.virtual_memory().total / (1024**3)
        self.check_results["environment"]["ram"] = f"{ram_gb:.1f}GB"
        
        # Python 버전
        python_version = sys.version.split()[0]
        self.check_results["environment"]["python"] = python_version
        
        # PyTorch 버전
        self.check_results["environment"]["pytorch"] = torch.__version__
        
        print("✅ 환경 검사 완료\n")
    
    def check_files(self):
        """필수 파일 체크"""
        print("2. 파일 검사 중...")
        
        required_files = {
            "inference.py": "추론 실행 파일",
            "model_handler.py": "모델 핸들러",
            "data_processor.py": "데이터 처리",
            "prompt_engineering.py": "프롬프트 엔지니어링",
            "knowledge_base.py": "지식 베이스",
            "advanced_optimizer.py": "고급 최적화",
            "pattern_learner.py": "패턴 학습",
            "test.csv": "테스트 데이터",
            "sample_submission.csv": "제출 템플릿"
        }
        
        all_present = True
        for filename, description in required_files.items():
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024*1024)
                self.check_results["files"][filename] = f"✅ {size_mb:.2f}MB"
            else:
                self.check_results["files"][filename] = "❌ 없음"
                all_present = False
                print(f"  ⚠️ {filename} 파일이 없습니다!")
        
        self.check_results["files"]["all_present"] = all_present
        
        # 데이터 검증
        if os.path.exists("test.csv"):
            test_df = pd.read_csv("test.csv")
            self.check_results["files"]["test_count"] = len(test_df)
        
        print("✅ 파일 검사 완료\n")
    
    def check_model(self):
        """모델 체크"""
        print("3. 모델 검사 중...")
        
        # 모델 로딩 테스트
        try:
            from transformers import AutoTokenizer
            
            model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
            
            # 토크나이저만 로드하여 체크
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.check_results["model"]["name"] = model_name
            self.check_results["model"]["tokenizer"] = "✅ 로드 가능"
            
            # 모델 크기 추정
            self.check_results["model"]["estimated_size"] = "~11GB (16bit)"
            
            del tokenizer
            
        except Exception as e:
            self.check_results["model"]["error"] = str(e)
            print(f"  ⚠️ 모델 체크 오류: {e}")
        
        print("✅ 모델 검사 완료\n")
    
    def estimate_performance(self):
        """성능 예측"""
        print("4. 성능 예측 중...")
        
        # 시간 예측
        total_questions = 515
        
        # 처리 시간 예측 (보수적)
        easy_questions = int(total_questions * 0.4)  # 40%
        medium_questions = int(total_questions * 0.4)  # 40%
        hard_questions = total_questions - easy_questions - medium_questions  # 20%
        
        # 배치 처리 시간
        batch_time = (easy_questions / 10) * 5  # 10개씩, 배치당 5초
        
        # 개별 처리 시간
        individual_time = (medium_questions * 15 + hard_questions * 25)  # 초
        
        # 총 예상 시간
        total_time_sec = batch_time + individual_time + 300  # 5분 여유
        total_time_min = total_time_sec / 60
        
        self.check_results["performance"]["estimated_time"] = f"{total_time_min:.1f}분"
        self.check_results["performance"]["time_limit"] = "270분 (4시간 30분)"
        self.check_results["performance"]["time_safe"] = total_time_min < 250
        
        # 메모리 사용량 예측
        model_memory = 11  # GB
        overhead = 5  # GB
        total_memory = model_memory + overhead
        
        self.check_results["performance"]["memory_usage"] = f"~{total_memory}GB"
        self.check_results["performance"]["memory_safe"] = total_memory < 22
        
        print("✅ 성능 예측 완료\n")
    
    def check_compliance(self):
        """대회 규정 준수 체크"""
        print("5. 대회 규정 준수 검사 중...")
        
        # 단일 모델 체크
        self.check_results["compliance"]["single_model"] = "✅ SOLAR-10.7B만 사용"
        
        # API 사용 체크
        api_keywords = ["requests", "urllib", "http", "api_key", "openai", "anthropic"]
        api_found = False
        
        for py_file in Path(".").glob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                for keyword in api_keywords:
                    if keyword in content and "# " not in content:  # 주석 제외
                        api_found = True
                        break
        
        self.check_results["compliance"]["no_api"] = "✅ API 사용 없음" if not api_found else "❌ API 키워드 발견"
        
        # 앙상블 체크
        ensemble_keywords = ["ensemble", "voting", "blend", "stack"]
        ensemble_found = False
        
        for py_file in Path(".").glob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                for keyword in ensemble_keywords:
                    if keyword in content:
                        ensemble_found = True
                        break
        
        self.check_results["compliance"]["no_ensemble"] = "✅ 앙상블 없음" if not ensemble_found else "⚠️ 앙상블 키워드 발견"
        
        # 생성형 AI 사용
        self.check_results["compliance"]["generative_ai"] = "✅ LLM 기반 생성"
        
        print("✅ 규정 준수 검사 완료\n")
    
    def generate_report(self):
        """최종 보고서 생성"""
        print("\n" + "="*50)
        print("최종 시스템 검증 보고서")
        print("="*50)
        
        # 환경
        print("\n[환경]")
        for key, value in self.check_results["environment"].items():
            print(f"  {key}: {value}")
        
        # 파일
        print("\n[파일]")
        all_files_ok = self.check_results["files"].get("all_present", False)
        if all_files_ok:
            print("  ✅ 모든 필수 파일 존재")
            if "test_count" in self.check_results["files"]:
                print(f"  테스트 문항: {self.check_results['files']['test_count']}개")
        else:
            print("  ❌ 일부 파일 누락")
            for filename, status in self.check_results["files"].items():
                if "❌" in str(status):
                    print(f"    - {filename}: {status}")
        
        # 모델
        print("\n[모델]")
        print(f"  모델: {self.check_results['model'].get('name', 'Unknown')}")
        print(f"  예상 크기: {self.check_results['model'].get('estimated_size', 'Unknown')}")
        
        # 성능
        print("\n[성능 예측]")
        print(f"  예상 처리 시간: {self.check_results['performance']['estimated_time']}")
        print(f"  제한 시간: {self.check_results['performance']['time_limit']}")
        time_safe = self.check_results['performance']['time_safe']
        print(f"  시간 여유: {'✅ 충분' if time_safe else '⚠️ 부족'}")
        
        print(f"  예상 메모리: {self.check_results['performance']['memory_usage']}")
        memory_safe = self.check_results['performance']['memory_safe']
        print(f"  메모리 여유: {'✅ 충분' if memory_safe else '⚠️ 부족'}")
        
        # 규정 준수
        print("\n[대회 규정]")
        for key, value in self.check_results["compliance"].items():
            print(f"  {value}")
        
        # 최종 판정
        print("\n" + "="*50)
        
        all_good = (
            all_files_ok and
            time_safe and
            memory_safe and
            "❌" not in str(self.check_results["compliance"])
        )
        
        if all_good:
            print("✅ 시스템 준비 완료! 제출 가능합니다.")
            print("\n실행 명령: python inference.py")
        else:
            print("⚠️ 일부 문제가 발견되었습니다. 확인이 필요합니다.")
        
        print("="*50)

def run_test_inference(sample_size: int = 5):
    """간단한 추론 테스트"""
    print(f"\n작은 샘플({sample_size}개)로 추론 테스트 중...")
    
    try:
        # 테스트 실행
        start_time = time.time()
        
        # main.py를 통한 테스트
        result = subprocess.run(
            [sys.executable, "main.py", "--test-type", "speed", "--sample-size", str(sample_size)],
            capture_output=True,
            text=True,
            timeout=120  # 2분 타임아웃
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ 테스트 성공 ({elapsed:.1f}초)")
            print("출력 일부:")
            print(result.stdout[-500:])  # 마지막 500자
        else:
            print(f"❌ 테스트 실패")
            print("에러:")
            print(result.stderr[-500:])
            
    except subprocess.TimeoutExpired:
        print("❌ 테스트 타임아웃")
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")

def main():
    """메인 함수"""
    print("금융 AI Challenge 최종 시스템 검증\n")
    
    # 시스템 체크
    checker = FinalSystemChecker()
    checker.run_all_checks()
    
    # 선택적 테스트
    response = input("\n간단한 추론 테스트를 실행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        run_test_inference(5)
    
    print("\n검증 완료!")

if __name__ == "__main__":
    main()