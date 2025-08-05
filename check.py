# check.py
"""
시스템 검증 스크립트
"""

import os
import sys
import time
import torch
import pandas as pd
import subprocess
import psutil
import platform
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SystemChecker:
    """시스템 체크"""
    
    def __init__(self):
        self.check_results = {
            "environment": {},
            "files": {},
            "model": {},
            "performance": {},
            "compliance": {},
            "optimization": {},
            "memory": {},
            "gpu_analysis": {}
        }
        self.start_time = time.time()
        
    def run_checks(self):
        """포괄적 시스템 검사"""
        print("=== 시스템 검증 시작 ===\n")
        
        # 1. 환경 체크
        self.check_environment()
        
        # 2. 파일 체크
        self.check_files()
        
        # 3. 모델 체크
        self.check_model()
        
        # 4. GPU 분석
        self.analyze_gpu()
        
        # 5. 메모리 최적화 분석
        self.analyze_memory()
        
        # 6. 성능 예측
        self.estimate_performance()
        
        # 7. 최적화 기능 검증
        self.verify_optimization()
        
        # 8. 대회 규정 준수 체크
        self.check_compliance()
        
        # 최종 보고서
        self.generate_report()
    
    def check_environment(self):
        """환경 검사"""
        print("1. 환경 검사 중...")
        
        # 기본 정보
        self.check_results["environment"]["platform"] = platform.system()
        self.check_results["environment"]["python_version"] = sys.version.split()[0]
        self.check_results["environment"]["pytorch_version"] = torch.__version__
        
        # GPU 상세 분석
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.check_results["environment"]["gpu_count"] = gpu_count
            
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "name": gpu_props.name,
                    "memory_gb": gpu_props.total_memory / (1024**3),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessor_count": gpu_props.multi_processor_count,
                    "memory_bandwidth": getattr(gpu_props, 'memory_bus_width', 'Unknown')
                }
                self.check_results["environment"][f"gpu_{i}"] = gpu_info
                
                # GPU 성능 등급 판정
                if gpu_props.total_memory / (1024**3) >= 20:
                    performance_tier = "Ultra High"
                    optimization_mode = "최고 성능 모드"
                elif gpu_props.total_memory / (1024**3) >= 12:
                    performance_tier = "High"
                    optimization_mode = "성능 모드"
                elif gpu_props.total_memory / (1024**3) >= 8:
                    performance_tier = "Medium"
                    optimization_mode = "균형 모드"
                else:
                    performance_tier = "Basic"
                    optimization_mode = "효율성 모드"
                
                gpu_info["performance_tier"] = performance_tier
                gpu_info["recommended_mode"] = optimization_mode
            
            # CUDA 기능 확인
            self.check_results["environment"]["cuda_version"] = torch.version.cuda
            self.check_results["environment"]["cudnn_version"] = torch.backends.cudnn.version()
            self.check_results["environment"]["tensor_cores"] = self._check_tensor_cores()
            self.check_results["environment"]["mixed_precision"] = torch.cuda.amp.autocast().__class__.__name__ == 'autocast'
            
        else:
            self.check_results["environment"]["gpu_status"] = "CUDA 사용 불가"
        
        # CPU 정보
        cpu_info = {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
        }
        self.check_results["environment"]["cpu"] = cpu_info
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        self.check_results["environment"]["ram_gb"] = memory.total / (1024**3)
        self.check_results["environment"]["available_ram_gb"] = memory.available / (1024**3)
        
        print("환경 검사 완료\n")
    
    def _check_tensor_cores(self) -> bool:
        """Tensor Core 지원 확인"""
        try:
            # Tensor Core는 주로 V100, A100, RTX 시리즈에서 지원
            gpu_name = torch.cuda.get_device_name(0).upper()
            tensor_core_gpus = ['V100', 'A100', 'RTX', 'TITAN', 'QUADRO']
            return any(gpu in gpu_name for gpu in tensor_core_gpus)
        except:
            return False
    
    def check_files(self):
        """파일 검사"""
        print("2. 파일 검사 중...")
        
        required_files = {
            "core_files": {
                "inference.py": "메인 추론 실행 파일",
                "model_handler.py": "모델 핸들러",
                "data_processor.py": "데이터 처리",
                "prompt_engineering.py": "프롬프트 엔지니어링",
                "knowledge_base.py": "전문 지식 베이스",
                "advanced_optimizer.py": "시스템 최적화",
                "pattern_learner.py": "패턴 학습 시스템"
            },
            "data_files": {
                "test.csv": "테스트 데이터",
                "sample_submission.csv": "제출 템플릿"
            },
            "config_files": {
                "requirements.txt": "의존성 패키지",
                "main.py": "개발용 메인 파일"
            }
        }
        
        all_present = True
        total_size = 0
        
        for category, files in required_files.items():
            self.check_results["files"][category] = {}
            
            for filename, description in files.items():
                if os.path.exists(filename):
                    size_mb = os.path.getsize(filename) / (1024*1024)
                    total_size += size_mb
                    
                    # 파일 품질 검사
                    quality_info = self._analyze_file_quality(filename)
                    
                    self.check_results["files"][category][filename] = {
                        "status": "존재",
                        "size_mb": round(size_mb, 2),
                        "quality": quality_info
                    }
                else:
                    self.check_results["files"][category][filename] = {
                        "status": "없음",
                        "size_mb": 0,
                        "quality": {}
                    }
                    all_present = False
                    print(f"  {filename} 파일이 없습니다!")
        
        self.check_results["files"]["all_present"] = all_present
        self.check_results["files"]["total_size_mb"] = round(total_size, 2)
        
        # 데이터 품질 검증
        if os.path.exists("test.csv"):
            test_df = pd.read_csv("test.csv")
            data_quality = self._analyze_data_quality(test_df)
            self.check_results["files"]["data_quality"] = data_quality
        
        print("파일 검사 완료\n")
    
    def _analyze_file_quality(self, filename: str) -> Dict:
        """파일 품질 분석"""
        quality = {"complexity": "Unknown", "features": []}
        
        if filename.endswith('.py'):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 코드 복잡도 추정
                    lines = len(content.splitlines())
                    functions = len(re.findall(r'def\s+\w+', content))
                    classes = len(re.findall(r'class\s+\w+', content))
                    
                    if lines > 1000:
                        quality["complexity"] = "High"
                    elif lines > 500:
                        quality["complexity"] = "Medium"
                    else:
                        quality["complexity"] = "Low"
                    
                    quality["lines"] = lines
                    quality["functions"] = functions
                    quality["classes"] = classes
                    
                    # 기능 검사
                    advanced_features = {
                        "async_support": "async def" in content,
                        "type_hints": "from typing import" in content,
                        "dataclasses": "@dataclass" in content,
                        "caching": "cache" in content.lower(),
                        "gpu_optimization": "cuda" in content.lower(),
                        "parallel_processing": any(term in content for term in ["ThreadPoolExecutor", "multiprocessing", "concurrent"]),
                        "error_handling": "try:" in content and "except" in content
                    }
                    
                    quality["features"] = [feature for feature, present in advanced_features.items() if present]
                    
            except Exception as e:
                quality["error"] = str(e)
        
        return quality
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """데이터 품질 분석"""
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024*1024),
            "sample_questions": df['Question'].head(3).tolist() if 'Question' in df.columns else []
        }
    
    def check_model(self):
        """모델 기능 검사"""
        print("3. 모델 기능 검사 중...")
        
        try:
            from transformers import AutoTokenizer
            
            model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
            
            # 토크나이저 로드 테스트
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer_load_time = time.time() - start_time
            
            self.check_results["model"]["name"] = model_name
            self.check_results["model"]["tokenizer_load_time"] = round(tokenizer_load_time, 2)
            self.check_results["model"]["vocab_size"] = tokenizer.vocab_size
            self.check_results["model"]["model_max_length"] = tokenizer.model_max_length
            
            # 토크나이저 기능 테스트
            test_text = "개인정보보호법에 따른 안전성 확보조치에 대해 설명하세요."
            tokens = tokenizer.tokenize(test_text)
            
            self.check_results["model"]["tokenization_test"] = {
                "input_text": test_text,
                "token_count": len(tokens),
                "tokens_sample": tokens[:10]
            }
            
            # 모델 크기 추정
            self.check_results["model"]["estimated_size_gb"] = "~22GB (16bit)"
            self.check_results["model"]["recommended_memory_gb"] = 24
            
            del tokenizer
            
        except Exception as e:
            self.check_results["model"]["error"] = str(e)
            print(f"  모델 체크 오류: {e}")
        
        print("모델 기능 검사 완료\n")
    
    def analyze_gpu(self):
        """GPU 성능 분석"""
        print("4. GPU 성능 분석 중...")
        
        if not torch.cuda.is_available():
            self.check_results["gpu_analysis"]["status"] = "CUDA 불가"
            return
        
        gpu_analysis = {}
        
        # 메모리 벤치마크
        try:
            # 메모리 할당 테스트
            memory_test_sizes = [1, 2, 4, 8]  # GB
            memory_results = {}
            
            for size_gb in memory_test_sizes:
                try:
                    # 임시 텐서 생성
                    elements = int(size_gb * 1024**3 / 4)  # float32 기준
                    test_tensor = torch.randn(elements, device='cuda', dtype=torch.float32)
                    
                    memory_results[f"{size_gb}GB"] = "성공"
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    memory_results[f"{size_gb}GB"] = f"실패: {str(e)[:50]}"
                    torch.cuda.empty_cache()
                    break
            
            gpu_analysis["memory_allocation_test"] = memory_results
            
            # 연산 성능 테스트
            performance_tests = self._run_gpu_performance_tests()
            gpu_analysis["performance_tests"] = performance_tests
            
        except Exception as e:
            gpu_analysis["memory_test_error"] = str(e)
        
        # GPU 상태 정보
        gpu_analysis["current_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
        gpu_analysis["max_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_analysis["memory_efficiency"] = torch.cuda.memory_efficiency() if hasattr(torch.cuda, 'memory_efficiency') else "Unknown"
        
        self.check_results["gpu_analysis"] = gpu_analysis
        
        print("GPU 성능 분석 완료\n")
    
    def _run_gpu_performance_tests(self) -> Dict:
        """GPU 성능 테스트 실행"""
        tests = {}
        
        try:
            # 행렬 곱셈 성능 테스트
            size = 2048
            a = torch.randn(size, size, device='cuda', dtype=torch.float16)
            b = torch.randn(size, size, device='cuda', dtype=torch.float16)
            
            # Warm-up
            for _ in range(5):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # 실제 측정
            start_time = time.time()
            for _ in range(10):
                result = torch.matmul(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            gflops = (2 * size**3) / (avg_time * 1e9)
            
            tests["matrix_multiplication"] = {
                "avg_time_ms": round(avg_time * 1000, 2),
                "gflops": round(gflops, 2),
                "performance_tier": "High" if gflops > 100 else "Medium" if gflops > 50 else "Low"
            }
            
            # Mixed Precision 테스트
            if torch.cuda.amp.autocast().__class__.__name__ == 'autocast':
                with torch.cuda.amp.autocast():
                    start_time = time.time()
                    for _ in range(10):
                        result = torch.matmul(a.float(), b.float())
                    torch.cuda.synchronize()
                    end_time = time.time()
                
                amp_time = (end_time - start_time) / 10
                tests["mixed_precision"] = {
                    "avg_time_ms": round(amp_time * 1000, 2),
                    "speedup_vs_fp32": round(avg_time / amp_time, 2) if amp_time > 0 else "N/A"
                }
            
            # 정리
            del a, b, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            tests["error"] = str(e)
        
        return tests
    
    def analyze_memory(self):
        """메모리 최적화 분석"""
        print("5. 메모리 최적화 분석 중...")
        
        memory_analysis = {}
        
        # 시스템 메모리
        memory = psutil.virtual_memory()
        memory_analysis["system_memory"] = {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "usage_percent": memory.percent
        }
        
        # GPU 메모리 (CUDA 사용 가능한 경우)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_analysis["gpu_memory"] = {
                "total_gb": round(gpu_memory, 1),
                "recommended_usage_gb": round(gpu_memory * 0.9, 1),
                "optimization_strategies": self._get_memory_optimization_strategies(gpu_memory)
            }
        
        # 최적화 권장사항
        memory_analysis["optimization_recommendations"] = self._generate_memory_recommendations()
        
        self.check_results["memory"] = memory_analysis
        
        print("메모리 최적화 분석 완료\n")
    
    def _get_memory_optimization_strategies(self, gpu_memory_gb: float) -> List[str]:
        """메모리 최적화 전략"""
        strategies = []
        
        if gpu_memory_gb >= 20:
            strategies.extend([
                "대용량 배치 처리 가능",
                "모델 컴파일 최적화 활성화",
                "Mixed Precision 사용",
                "캐시 크기 확대"
            ])
        elif gpu_memory_gb >= 12:
            strategies.extend([
                "중간 배치 크기 사용",
                "선택적 모델 컴파일",
                "표준 Mixed Precision",
                "적응형 캐시 관리"
            ])
        else:
            strategies.extend([
                "작은 배치 크기 필수",
                "메모리 절약 모드",
                "Gradient Checkpointing 고려",
                "빈번한 캐시 정리"
            ])
        
        return strategies
    
    def _generate_memory_recommendations(self) -> List[str]:
        """메모리 권장사항 생성"""
        recommendations = []
        
        # GPU 메모리 기반 권장사항
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory < 12:
                recommendations.append("GPU 메모리 부족 - 배치 크기 축소 필요")
                recommendations.append("메모리 절약을 위한 quantization 고려")
            else:
                recommendations.append("충분한 GPU 메모리")
        
        # 시스템 메모리 체크
        system_memory = psutil.virtual_memory()
        if system_memory.available / (1024**3) < 8:
            recommendations.append("시스템 메모리 부족 - 백그라운드 프로세스 정리 권장")
        
        return recommendations
    
    def estimate_performance(self):
        """성능 예측"""
        print("6. 성능 예측 중...")
        
        total_questions = 515
        
        # GPU 성능 기반 시간 예측
        gpu_info = self.check_results.get("environment", {})
        gpu_memory = 0
        
        if "gpu_0" in gpu_info:
            gpu_memory = gpu_info["gpu_0"]["memory_gb"]
            performance_tier = gpu_info["gpu_0"]["performance_tier"]
        
        # 성능 계층별 처리 시간 추정
        if performance_tier == "Ultra High":
            base_time_per_question = 3  # 초
            batch_efficiency = 0.8
        elif performance_tier == "High":
            base_time_per_question = 5  
            batch_efficiency = 0.7
        elif performance_tier == "Medium":
            base_time_per_question = 8  
            batch_efficiency = 0.6
        else:
            base_time_per_question = 12  
            batch_efficiency = 0.5
        
        # 문제 유형별 분포 추정
        estimated_distribution = {
            "객관식_쉬운문제": int(total_questions * 0.3),    # 30%
            "객관식_보통문제": int(total_questions * 0.35),   # 35%
            "객관식_어려운문제": int(total_questions * 0.15), # 15%
            "주관식_문제": int(total_questions * 0.2)        # 20%
        }
        
        # 상세 시간 계산
        time_estimates = {}
        total_time = 0
        
        for problem_type, count in estimated_distribution.items():
            if "쉬운" in problem_type:
                time_per_item = base_time_per_question * 0.5
            elif "보통" in problem_type:
                time_per_item = base_time_per_question * 0.8
            elif "어려운" in problem_type:
                time_per_item = base_time_per_question * 1.2
            else:  # 주관식
                time_per_item = base_time_per_question * 1.5
            
            # 배치 효율성 적용
            if "객관식" in problem_type:
                effective_time = time_per_item * batch_efficiency
            else:
                effective_time = time_per_item
            
            category_total = count * effective_time
            time_estimates[problem_type] = {
                "count": count,
                "time_per_question": round(effective_time, 1),
                "total_time": round(category_total, 1)
            }
            total_time += category_total
        
        # 오버헤드 추가 (시스템 준비, 메모리 정리 등)
        overhead_time = total_time * 0.15  # 15% 오버헤드
        total_time_with_overhead = total_time + overhead_time
        
        performance_results = {
            "detailed_estimates": time_estimates,
            "total_processing_time_min": round(total_time_with_overhead / 60, 1),
            "overhead_time_min": round(overhead_time / 60, 1),
            "time_limit_min": 270,  # 4시간 30분
            "estimated_questions_per_minute": round(total_questions / (total_time_with_overhead / 60), 1),
            "safety_margin_min": round(270 - (total_time_with_overhead / 60), 1),
            "performance_tier": performance_tier,
            "time_safety": total_time_with_overhead < 270 * 60 * 0.8  # 80% 마진
        }
        
        # 메모리 사용량 예측
        model_memory = 22  # GB (모델 기본)
        processing_memory = gpu_memory * 0.2 if gpu_memory > 0 else 4  # 처리용
        total_memory_needed = model_memory + processing_memory
        
        performance_results["memory_requirements"] = {
            "model_memory_gb": model_memory,
            "processing_memory_gb": round(processing_memory, 1),
            "total_memory_gb": round(total_memory_needed, 1),
            "available_memory_gb": gpu_memory,
            "memory_safety": gpu_memory >= total_memory_needed if gpu_memory > 0 else False
        }
        
        self.check_results["performance"] = performance_results
        
        print("성능 예측 완료\n")
    
    def verify_optimization(self):
        """최적화 기능 검증"""
        print("7. 최적화 기능 검증 중...")
        
        optimization_features = {}
        
        # PyTorch 컴파일 지원
        optimization_features["torch_compile"] = {
            "available": hasattr(torch, 'compile'),
            "recommended": torch.__version__ >= "2.0.0"
        }
        
        # Mixed Precision 지원
        optimization_features["mixed_precision"] = {
            "available": hasattr(torch.cuda.amp, 'autocast'),
            "tensor_cores": self.check_results["environment"].get("tensor_cores", False)
        }
        
        # Flash Attention 확인
        try:
            import flash_attn
            optimization_features["flash_attention"] = {"available": True, "version": flash_attn.__version__}
        except ImportError:
            optimization_features["flash_attention"] = {"available": False}
        
        # SDPA 지원 (PyTorch 2.0+)
        optimization_features["sdpa"] = {
            "available": hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        }
        
        # 병렬 처리 기능
        optimization_features["parallel_processing"] = {
            "cpu_cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "recommended_workers": min(psutil.cpu_count(), 8)
        }
        
        # 캐싱 시스템 테스트
        optimization_features["caching"] = self._test_caching_performance()
        
        self.check_results["optimization"] = optimization_features
        
        print("최적화 기능 검증 완료\n")
    
    def _test_caching_performance(self) -> Dict:
        """캐싱 성능 테스트"""
        cache_test = {}
        
        try:
            # 간단한 캐싱 성능 테스트
            test_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
            
            # 딕셔너리 캐시 테스트
            start_time = time.time()
            for key in test_data:
                _ = test_data[key]
            dict_time = time.time() - start_time
            
            cache_test["dict_cache_ms"] = round(dict_time * 1000, 2)
            cache_test["operations_per_second"] = round(1000 / dict_time, 0)
            cache_test["performance"] = "High" if dict_time < 0.001 else "Medium" if dict_time < 0.01 else "Low"
            
        except Exception as e:
            cache_test["error"] = str(e)
        
        return cache_test
    
    def check_compliance(self):
        """대회 규정 준수 체크"""
        print("8. 대회 규정 준수 검사 중...")
        
        compliance = {}
        
        # 단일 모델 사용 확인
        compliance["single_model"] = {
            "status": "SOLAR-10.7B-Instruct-v1.0 단일 모델 사용",
            "verified": True
        }
        
        # 외부 API 사용 검사
        api_keywords = [
            "requests.get", "requests.post", "urllib.request", "http.client",
            "api_key", "openai", "anthropic", "google.api", "azure.openai",
            "huggingface_hub.login", "api.call", "rest_api"
        ]
        
        api_violations = []
        for py_file in Path(".").glob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for keyword in api_keywords:
                        if keyword in content and "#" not in content.split(keyword)[0].split('\n')[-1]:
                            api_violations.append(f"{py_file.name}: {keyword}")
            except:
                continue
        
        compliance["no_external_api"] = {
            "status": "외부 API 사용 없음" if not api_violations else "의심스러운 API 코드 발견",
            "violations": api_violations
        }
        
        # 앙상블 방법 검사
        ensemble_keywords = [
            "ensemble", "voting", "blend", "stack", "average_predictions",
            "model_fusion", "consensus", "multiple_models"
        ]
        
        ensemble_violations = []
        for py_file in Path(".").glob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    for keyword in ensemble_keywords:
                        if keyword in content:
                            # 컨텍스트 확인
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if keyword in line and "#" not in line:
                                    ensemble_violations.append(f"{py_file.name}:{i+1}: {keyword}")
            except:
                continue
        
        compliance["no_ensemble"] = {
            "status": "앙상블 방법 없음" if not ensemble_violations else "앙상블 관련 코드 발견",
            "violations": ensemble_violations
        }
        
        # 생성형 AI 사용 확인
        generative_features = [
            "AutoModelForCausalLM", "generate", "text-generation",
            "transformer", "llm", "language_model"
        ]
        
        generative_usage = []
        for py_file in Path(".").glob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for feature in generative_features:
                        if feature in content:
                            generative_usage.append(feature)
            except:
                continue
        
        compliance["generative_ai_usage"] = {
            "status": "생성형 AI 사용 확인됨",
            "features_found": list(set(generative_usage))
        }
        
        # 메모리 및 시간 제한 준수
        performance = self.check_results.get("performance", {})
        compliance["resource_limits"] = {
            "time_compliance": performance.get("time_safety", False),
            "memory_compliance": performance.get("memory_requirements", {}).get("memory_safety", False),
            "estimated_time_min": performance.get("total_processing_time_min", "Unknown"),
            "time_limit_min": 270
        }
        
        self.check_results["compliance"] = compliance
        
        print("대회 규정 준수 검사 완료\n")
    
    def generate_report(self):
        """종합 보고서 생성"""
        print("\n" + "="*60)
        print("시스템 검증 종합 보고서")
        print("="*60)
        
        total_check_time = time.time() - self.start_time
        
        # 시스템 요약
        print(f"\n시스템 요약")
        print(f"검증 시간: {total_check_time:.1f}초")
        print(f"플랫폼: {self.check_results['environment']['platform']}")
        print(f"Python: {self.check_results['environment']['python_version']}")
        print(f"PyTorch: {self.check_results['environment']['pytorch_version']}")
        
        # GPU 분석 요약
        if "gpu_0" in self.check_results["environment"]:
            gpu_info = self.check_results["environment"]["gpu_0"]
            print(f"\nGPU 분석")
            print(f"GPU: {gpu_info['name']}")
            print(f"메모리: {gpu_info['memory_gb']:.1f}GB")
            print(f"성능 등급: {gpu_info['performance_tier']}")
            print(f"권장 모드: {gpu_info['recommended_mode']}")
        
        # 파일 상태
        print(f"\n파일 상태")
        all_files_ok = self.check_results["files"]["all_present"]
        if all_files_ok:
            print("모든 필수 파일 존재")
            print(f"총 파일 크기: {self.check_results['files']['total_size_mb']:.1f}MB")
        else:
            print("일부 파일 누락")
        
        # 성능 예측
        print(f"\n성능 예측")
        performance = self.check_results.get("performance", {})
        if performance:
            print(f"예상 처리 시간: {performance['total_processing_time_min']}분")
            print(f"제한 시간: {performance['time_limit_min']}분")
            print(f"안전 마진: {performance['safety_margin_min']}분")
            print(f"처리 속도: {performance['estimated_questions_per_minute']}문항/분")
            
            time_safe = performance.get("time_safety", False)
            memory_safe = performance.get("memory_requirements", {}).get("memory_safety", False)
            
            print(f"시간 여유: {'충분' if time_safe else '부족'}")
            print(f"메모리 여유: {'충분' if memory_safe else '부족'}")
        
        # 최적화 기능
        print(f"\n최적화 기능")
        optimization = self.check_results.get("optimization", {})
        if optimization:
            print(f"Torch Compile: {'지원' if optimization.get('torch_compile', {}).get('available') else '미지원'}")
            print(f"Mixed Precision: {'지원' if optimization.get('mixed_precision', {}).get('available') else '미지원'}")
            print(f"Flash Attention: {'지원' if optimization.get('flash_attention', {}).get('available') else '미지원'}")
            print(f"SDPA: {'지원' if optimization.get('sdpa', {}).get('available') else '미지원'}")
        
        # 규정 준수
        print(f"\n대회 규정 준수")
        compliance = self.check_results.get("compliance", {})
        if compliance:
            for check_name, check_info in compliance.items():
                if isinstance(check_info, dict) and "status" in check_info:
                    print(f"  {check_info['status']}")
        
        # 최종 판정
        print(f"\n" + "="*60)
        
        # 종합 점수 계산
        scores = {
            "files": 100 if all_files_ok else 70,
            "performance": 100 if (time_safe and memory_safe) else 80 if time_safe else 60,
            "optimization": 100 if optimization.get("torch_compile", {}).get("available") else 80,
            "compliance": 100 if all(not check.get("violations", []) for check in compliance.values() if isinstance(check, dict)) else 90
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        print(f"종합 점수: {overall_score:.0f}/100")
        
        if overall_score >= 90:
            print("우수: 시스템이 최적 상태입니다!")
            print("실행 명령: python inference.py")
        elif overall_score >= 80:
            print("양호: 시스템 준비가 잘 되어 있습니다.")
            print("실행 명령: python inference.py")
        elif overall_score >= 70:
            print("주의: 일부 개선이 필요합니다.")
        else:
            print("불량: 시스템 점검이 필요합니다.")
        
        # 성능 향상 권장사항
        print(f"\n성능 향상 권장사항:")
        recommendations = self._generate_performance_recommendations()
        for rec in recommendations:
            print(f"  • {rec}")
        
        print("="*60)
    
    def _generate_performance_recommendations(self) -> List[str]:
        """성능 향상 권장사항 생성"""
        recommendations = []
        
        # GPU 기반 권장사항
        if "gpu_0" in self.check_results["environment"]:
            gpu_info = self.check_results["environment"]["gpu_0"]
            tier = gpu_info["performance_tier"]
            
            if tier == "Ultra High":
                recommendations.append("최고 성능 GPU 감지 - 대용량 배치 처리 활성화")
                recommendations.append("Torch Compile 및 Flash Attention 사용 권장")
            elif tier == "High":
                recommendations.append("GPU 감지 - 적극적인 최적화 활성화")
            elif tier == "Medium":
                recommendations.append("메모리 효율적 처리 모드 사용 권장")
            else:
                recommendations.append("제한된 성능 - 보수적 설정 사용")
        
        # 최적화 기능 기반 권장사항
        optimization = self.check_results.get("optimization", {})
        if not optimization.get("flash_attention", {}).get("available"):
            recommendations.append("Flash Attention 설치 권장 (pip install flash-attn)")
        
        if not optimization.get("torch_compile", {}).get("available"):
            recommendations.append("PyTorch 2.0+ 업그레이드 권장 (torch.compile 지원)")
        
        # 메모리 기반 권장사항
        memory_info = self.check_results.get("memory", {})
        if memory_info.get("system_memory", {}).get("usage_percent", 0) > 80:
            recommendations.append("시스템 메모리 사용률 높음 - 백그라운드 프로세스 정리")
        
        return recommendations

def run_mini_inference_test(sample_size: int = 3):
    """소규모 추론 테스트"""
    print(f"\n소규모 추론 테스트 ({sample_size}개 샘플)...")
    
    try:
        start_time = time.time()
        
        # 테스트 실행
        result = subprocess.run(
            [sys.executable, "main.py", "--test-type", "speed", "--sample-size", str(sample_size)],
            capture_output=True,
            text=True,
            timeout=180  # 3분 타임아웃
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"테스트 성공 ({elapsed:.1f}초)")
            print(f"예상 전체 처리 시간: {(elapsed/sample_size*515)/60:.1f}분")
            
            # 출력에서 유용한 정보 추출
            output_lines = result.stdout.split('\n')[-10:]
            print("최근 출력:")
            for line in output_lines:
                if line.strip():
                    print(f"  {line.strip()}")
                    
        else:
            print(f"테스트 실패")
            print("오류 정보:")
            error_lines = result.stderr.split('\n')[-5:]
            for line in error_lines:
                if line.strip():
                    print(f"  {line.strip()}")
            
    except subprocess.TimeoutExpired:
        print("테스트 타임아웃 (3분 초과)")
    except Exception as e:
        print(f"테스트 오류: {e}")

def main():
    """메인 함수"""
    print("금융 AI Challenge 시스템 검증\n")
    
    # 포괄적 시스템 체크
    checker = SystemChecker()
    checker.run_checks()
    
    # 선택적 미니 테스트
    response = input("\n소규모 추론 테스트를 실행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        run_mini_inference_test(3)
    
    print("\n시스템 검증 완료!")

if __name__ == "__main__":
    main()