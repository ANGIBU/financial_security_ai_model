# model_handler.py

"""
모델 핸들러 (강화버전)
- LLM 모델 로딩 및 관리
- 텍스트 생성 및 추론
- 한국어 최적화 설정
- 메모리 관리
- 다단계 추론 지원
- 신뢰도 보정 시스템
- 동적 생성 파라미터 조정
- 에러 복구 및 대안 전략
"""

import torch
import re
import time
import gc
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
import warnings
warnings.filterwarnings("ignore")

@dataclass
class InferenceResult:
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int
    inference_time: float = 0.0
    korean_quality: float = 0.0
    generation_attempts: int = 1
    fallback_used: bool = False
    confidence_calibrated: bool = False

@dataclass
class GenerationStrategy:
    name: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    confidence_threshold: float
    success_rate: float = 0.0
    usage_count: int = 0

class ModelHandler:
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = True, max_memory_gb: int = 22, verbose: bool = False):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        if self.verbose:
            print(f"모델 로딩: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if self.verbose:
                print("Flash Attention 2 활성화")
        except ImportError:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model_kwargs["attn_implementation"] = "sdpa"
                if self.verbose:
                    print("SDPA Attention 활성화")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                if self.verbose:
                    print("Torch Compile 활성화")
            except Exception:
                if self.verbose:
                    print("Torch Compile 실패, 기본 모델 사용")
        
        self._prepare_korean_optimization()
        
        self.response_cache = {}
        self.cache_hits = 0
        self.max_cache_size = 400
        self.memory_cleanup_counter = 0
        
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "korean_quality_scores": [],
            "avg_inference_time": 0.0,
            "cache_efficiency": 0.0,
            "generation_failures": 0,
            "timeout_failures": 0,
            "strategy_usage": {},
            "confidence_calibrations": 0,
            "fallback_activations": 0
        }
        
        # 고급 기능 추가
        self.generation_strategies = self._initialize_generation_strategies()
        self.confidence_calibrator = self._initialize_confidence_calibrator()
        self.error_recovery = self._initialize_error_recovery()
        self.adaptive_parameters = self._initialize_adaptive_parameters()
        
        # 성능 모니터링
        self.performance_monitor = {
            "strategy_performance": {},
            "quality_trends": [],
            "confidence_trends": [],
            "inference_times": []
        }
        
        # 메타 학습
        self.meta_learning = {
            "successful_configs": [],
            "failed_configs": [],
            "optimal_strategy": "balanced",
            "adaptation_counter": 0
        }
        
        if self.verbose:
            print("모델 로딩 완료")
    
    def _initialize_generation_strategies(self) -> Dict[str, GenerationStrategy]:
        return {
            "conservative": GenerationStrategy(
                name="conservative",
                temperature=0.3,
                top_p=0.7,
                top_k=20,
                max_tokens=250,
                repetition_penalty=1.05,
                confidence_threshold=0.8
            ),
            "balanced": GenerationStrategy(
                name="balanced",
                temperature=0.6,
                top_p=0.85,
                top_k=35,
                max_tokens=280,
                repetition_penalty=1.05,
                confidence_threshold=0.6
            ),
            "creative": GenerationStrategy(
                name="creative",
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                max_tokens=320,
                repetition_penalty=1.02,
                confidence_threshold=0.5
            ),
            "precise": GenerationStrategy(
                name="precise",
                temperature=0.2,
                top_p=0.6,
                top_k=15,
                max_tokens=200,
                repetition_penalty=1.1,
                confidence_threshold=0.85
            ),
            "adaptive": GenerationStrategy(
                name="adaptive",
                temperature=0.5,
                top_p=0.8,
                top_k=30,
                max_tokens=280,
                repetition_penalty=1.03,
                confidence_threshold=0.7
            )
        }
    
    def _initialize_confidence_calibrator(self) -> Dict:
        return {
            "calibration_history": [],
            "confidence_mapping": {
                "very_high": (0.9, 1.0),
                "high": (0.7, 0.9),
                "medium": (0.5, 0.7),
                "low": (0.3, 0.5),
                "very_low": (0.0, 0.3)
            },
            "adjustment_factors": {
                "question_complexity": 1.0,
                "domain_familiarity": 1.0,
                "generation_quality": 1.0,
                "historical_accuracy": 1.0
            }
        }
    
    def _initialize_error_recovery(self) -> Dict:
        return {
            "recovery_strategies": {
                "generation_failure": "retry_with_different_params",
                "low_quality": "apply_post_processing",
                "timeout": "use_fallback_response",
                "memory_error": "reduce_context_length"
            },
            "fallback_responses": {
                "multiple_choice": lambda: str(random.randint(1, 5)),
                "subjective": lambda: self._get_fallback_subjective_answer()
            },
            "recovery_attempts": 0,
            "success_rate": 0.0
        }
    
    def _initialize_adaptive_parameters(self) -> Dict:
        return {
            "dynamic_adjustment": True,
            "performance_threshold": 0.7,
            "adaptation_rate": 0.1,
            "parameter_history": [],
            "optimal_ranges": {
                "temperature": (0.2, 0.8),
                "top_p": (0.6, 0.95),
                "top_k": (10, 60)
            }
        }
    
    def _prepare_korean_optimization(self):
        self.bad_words_ids = []
        
        problematic_patterns = [
            "金融", "交易", "安全", "管理", "個人", "資訊", "電子", "系統",
            "保護", "認證", "加密", "網路", "軟件", "硬件", "软件", "个人",
            "资讯", "电子", "系统", "保护", "认证", "网络"
        ]
        
        for pattern in problematic_patterns:
            try:
                tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
                if tokens and len(tokens) <= 3:
                    self.bad_words_ids.append(tokens)
            except:
                continue
        
        special_symbols = ["①", "②", "③", "④", "⑤", "➀", "➁", "❶", "❷", "❸", "bo", "Bo", "BO"]
        for symbol in special_symbols:
            try:
                tokens = self.tokenizer.encode(symbol, add_special_tokens=False)
                if tokens:
                    self.bad_words_ids.append(tokens)
            except:
                continue
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str) -> str:
        if question_type == "multiple_choice":
            korean_prefix = "다음 금융보안 문제의 정답 번호를 정확히 선택하세요.\n\n"
            korean_suffix = "\n\n정답은 1, 2, 3, 4, 5 중 하나의 번호입니다.\n정답:"
        else:
            korean_prefix = "다음 금융보안 질문에 한국어로 전문적인 답변을 작성하세요.\n\n"
            korean_suffix = "\n\n한국어로 답변:"
        
        return korean_prefix + prompt + korean_suffix
    
    def select_optimal_strategy(self, question_type: str, complexity: float, 
                              target_confidence: float) -> GenerationStrategy:
        """최적 생성 전략 선택"""
        
        # 전략별 성능 점수 계산
        strategy_scores = {}
        
        for name, strategy in self.generation_strategies.items():
            score = 0.0
            
            # 기본 성공률
            if strategy.usage_count > 0:
                score += strategy.success_rate * 0.4
            else:
                score += 0.5  # 기본 점수
            
            # 목표 신뢰도와의 일치성
            confidence_diff = abs(strategy.confidence_threshold - target_confidence)
            score += (1 - confidence_diff) * 0.3
            
            # 복잡도 적합성
            if complexity > 0.7 and name in ["precise", "conservative"]:
                score += 0.2
            elif complexity < 0.3 and name in ["creative", "balanced"]:
                score += 0.2
            
            # 질문 유형 적합성
            if question_type == "multiple_choice" and name in ["precise", "conservative"]:
                score += 0.1
            elif question_type == "subjective" and name in ["balanced", "creative"]:
                score += 0.1
            
            strategy_scores[name] = score
        
        # 최고 점수 전략 선택
        best_strategy_name = max(strategy_scores.items(), key=lambda x: x[1])[0]
        selected_strategy = self.generation_strategies[best_strategy_name]
        
        # 적응적 조정
        if self.adaptive_parameters["dynamic_adjustment"]:
            selected_strategy = self._adapt_strategy_parameters(selected_strategy, complexity)
        
        return selected_strategy
    
    def _adapt_strategy_parameters(self, strategy: GenerationStrategy, complexity: float) -> GenerationStrategy:
        """전략 파라미터 적응적 조정"""
        
        adapted_strategy = GenerationStrategy(
            name=f"{strategy.name}_adapted",
            temperature=strategy.temperature,
            top_p=strategy.top_p,
            top_k=strategy.top_k,
            max_tokens=strategy.max_tokens,
            repetition_penalty=strategy.repetition_penalty,
            confidence_threshold=strategy.confidence_threshold
        )
        
        # 복잡도 기반 조정
        if complexity > 0.7:
            # 높은 복잡도 - 더 보수적으로
            adapted_strategy.temperature *= 0.8
            adapted_strategy.top_p *= 0.9
            adapted_strategy.top_k = int(adapted_strategy.top_k * 0.8)
        elif complexity < 0.3:
            # 낮은 복잡도 - 더 창의적으로
            adapted_strategy.temperature *= 1.2
            adapted_strategy.top_p *= 1.05
            adapted_strategy.top_k = int(adapted_strategy.top_k * 1.2)
        
        # 범위 제한
        optimal_ranges = self.adaptive_parameters["optimal_ranges"]
        adapted_strategy.temperature = max(optimal_ranges["temperature"][0], 
                                         min(adapted_strategy.temperature, optimal_ranges["temperature"][1]))
        adapted_strategy.top_p = max(optimal_ranges["top_p"][0], 
                                   min(adapted_strategy.top_p, optimal_ranges["top_p"][1]))
        adapted_strategy.top_k = max(optimal_ranges["top_k"][0], 
                                   min(adapted_strategy.top_k, optimal_ranges["top_k"][1]))
        
        return adapted_strategy
    
    def generate_response_enhanced(self, prompt: str, question_type: str,
                                 max_attempts: int = 3, target_confidence: float = 0.7,
                                 complexity: float = 0.5) -> InferenceResult:
        """향상된 응답 생성"""
        
        start_time = time.time()
        
        cache_key = hash(prompt[:100] + question_type + str(target_confidence))
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            cached.inference_time = 0.01
            return cached
        
        optimized_prompt = self._create_korean_optimized_prompt(prompt, question_type)
        
        best_result = None
        best_score = 0
        generation_errors = 0
        
        for attempt in range(max_attempts):
            try:
                # 전략 선택
                strategy = self.select_optimal_strategy(question_type, complexity, target_confidence)
                
                # 생성 설정 구성
                gen_config = GenerationConfig(
                    do_sample=True,
                    temperature=strategy.temperature,
                    top_p=strategy.top_p,
                    top_k=strategy.top_k,
                    max_new_tokens=strategy.max_tokens,
                    repetition_penalty=strategy.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=self.bad_words_ids[:15] if self.bad_words_ids else None
                )
                
                inputs = self.tokenizer(
                    optimized_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1000
                ).to(self.model.device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        try:
                            outputs = self.model.generate(
                                **inputs,
                                generation_config=gen_config
                            )
                        except Exception as gen_error:
                            generation_errors += 1
                            if self.verbose:
                                print(f"생성 오류 (시도 {attempt+1}): {str(gen_error)[:100]}")
                            
                            # 에러 복구 시도
                            recovery_result = self._attempt_error_recovery(gen_error, optimized_prompt, question_type)
                            if recovery_result:
                                outputs = recovery_result
                            else:
                                continue
                
                raw_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                cleaned_response = self._clean_korean_text_enhanced(raw_response)
                
                # 결과 평가
                result = self._evaluate_generation_result(
                    cleaned_response, question_type, strategy, attempt
                )
                
                # 신뢰도 보정
                calibrated_result = self._calibrate_confidence(result, prompt, question_type, complexity)
                
                # 품질 점수 계산
                quality_score = self._calculate_overall_quality_score(calibrated_result)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_result = calibrated_result
                
                # 전략 성능 업데이트
                self._update_strategy_performance(strategy, quality_score > 0.7)
                
                # 목표 품질 달성 시 조기 종료
                if quality_score > 0.8:
                    break
                    
            except Exception as e:
                generation_errors += 1
                if self.verbose:
                    print(f"전체 생성 오류 (시도 {attempt+1}): {str(e)[:100]}")
                continue
        
        # 최종 결과 처리
        if best_result is None:
            best_result = self._create_fallback_result(question_type)
            best_result.fallback_used = True
            self.generation_stats["fallback_activations"] += 1
        
        best_result.inference_time = time.time() - start_time
        best_result.generation_attempts = max_attempts
        
        # 캐시 저장
        self._manage_cache()
        self.response_cache[cache_key] = best_result
        
        # 통계 업데이트
        self._update_generation_stats(best_result, best_result is not None and not best_result.fallback_used)
        
        # 메모리 정리
        self._perform_memory_cleanup()
        
        return best_result
    
    def _attempt_error_recovery(self, error: Exception, prompt: str, question_type: str):
        """에러 복구 시도"""
        
        self.error_recovery["recovery_attempts"] += 1
        
        error_type = type(error).__name__
        
        if "memory" in str(error).lower() or "cuda" in str(error).lower():
            # 메모리 오류 - 컨텍스트 길이 줄이기
            shortened_prompt = prompt[:len(prompt)//2]
            try:
                inputs = self.tokenizer(
                    shortened_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=500
                ).to(self.model.device)
                
                gen_config = GenerationConfig(
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    max_new_tokens=100,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, generation_config=gen_config)
                
                return outputs
                
            except:
                return None
        
        return None
    
    def _calculate_overall_quality_score(self, result: InferenceResult) -> float:
        """전체 품질 점수 계산"""
        
        score = 0.0
        
        # 기본 신뢰도 (40%)
        score += result.confidence * 0.4
        
        # 한국어 품질 (30%)
        score += result.korean_quality * 0.3
        
        # 추론 품질 (20%)
        score += result.reasoning_quality * 0.2
        
        # 분석 깊이 (10%)
        depth_score = min(result.analysis_depth / 5, 1.0)
        score += depth_score * 0.1
        
        return min(score, 1.0)
    
    def _calibrate_confidence(self, result: InferenceResult, prompt: str, 
                            question_type: str, complexity: float) -> InferenceResult:
        """신뢰도 보정"""
        
        calibrator = self.confidence_calibrator
        adjustment_factors = calibrator["adjustment_factors"]
        
        # 질문 복잡도 조정
        complexity_adjustment = 1.0 - (complexity * 0.2)
        
        # 도메인 친숙도 조정 (간단한 휴리스틱)
        domain_keywords = ["개인정보", "전자금융", "정보보안", "사이버보안"]
        domain_familiarity = sum(1 for keyword in domain_keywords if keyword in prompt.lower()) / len(domain_keywords)
        domain_adjustment = 0.8 + (domain_familiarity * 0.2)
        
        # 생성 품질 조정
        generation_quality = result.korean_quality * result.reasoning_quality
        quality_adjustment = 0.9 + (generation_quality * 0.1)
        
        # 역사적 정확도 조정 (간략화)
        historical_adjustment = 1.0
        
        # 최종 조정
        total_adjustment = (complexity_adjustment * domain_adjustment * 
                          quality_adjustment * historical_adjustment)
        
        calibrated_confidence = result.confidence * total_adjustment
        calibrated_confidence = max(0.1, min(calibrated_confidence, 0.95))
        
        result.confidence = calibrated_confidence
        result.confidence_calibrated = True
        
        self.generation_stats["confidence_calibrations"] += 1
        
        return result
    
    def _update_strategy_performance(self, strategy: GenerationStrategy, success: bool):
        """전략 성능 업데이트"""
        
        strategy.usage_count += 1
        
        if success:
            strategy.success_rate = ((strategy.success_rate * (strategy.usage_count - 1)) + 1.0) / strategy.usage_count
        else:
            strategy.success_rate = (strategy.success_rate * (strategy.usage_count - 1)) / strategy.usage_count
        
        # 성능 모니터링 업데이트
        if strategy.name not in self.performance_monitor["strategy_performance"]:
            self.performance_monitor["strategy_performance"][strategy.name] = []
        
        self.performance_monitor["strategy_performance"][strategy.name].append({
            "success": success,
            "timestamp": time.time(),
            "success_rate": strategy.success_rate
        })
        
        # 최근 10개 기록만 유지
        if len(self.performance_monitor["strategy_performance"][strategy.name]) > 10:
            self.performance_monitor["strategy_performance"][strategy.name] = \
                self.performance_monitor["strategy_performance"][strategy.name][-10:]
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 3) -> InferenceResult:
        """기본 응답 생성 (하위 호환성)"""
        return self.generate_response_enhanced(prompt, question_type, max_attempts)
    
    def _extract_mc_answer_enhanced(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.strip()
        
        if re.match(r'^[1-5]$', text):
            return text
        
        priority_patterns = [
            (r'정답[:\s]*([1-5])', 0.9),
            (r'답[:\s]*([1-5])', 0.85),
            (r'^([1-5])\s*$', 0.9),
            (r'^([1-5])\s*번', 0.8),
            (r'선택[:\s]*([1-5])', 0.8),
            (r'([1-5])번이', 0.75),
            (r'([1-5])가\s*정답', 0.75),
            (r'([1-5])이\s*정답', 0.75),
            (r'([1-5])\s*이\s*적절', 0.7),
            (r'([1-5])\s*가\s*적절', 0.7),
            (r'따라서\s*([1-5])', 0.7),
            (r'결론적으로\s*([1-5])', 0.7)
        ]
        
        best_match = None
        best_confidence = 0
        
        for pattern, confidence in priority_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches and confidence > best_confidence:
                answer = matches[0]
                if answer.isdigit() and 1 <= int(answer) <= 5:
                    best_match = answer
                    best_confidence = confidence
        
        if best_match:
            return best_match
        
        numbers = re.findall(r'[1-5]', text)
        if numbers:
            return numbers[0]
        
        return ""
    
    def _clean_korean_text_enhanced(self, text: str) -> str:
        if not text:
            return ""
        
        original_text = text
        original_length = len(text)
        
        # 기본 정리
        text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)
        
        safe_replacements = {
            '金融': '금융', '交易': '거래', '安全': '안전', '管理': '관리',
            '個人': '개인', '資訊': '정보', '電子': '전자', '系統': '시스템',
            '保護': '보호', '認證': '인증', '加密': '암호화', '網路': '네트워크',
            '軟件': '소프트웨어', '硬件': '하드웨어', '软件': '소프트웨어',
            '个人': '개인', '资讯': '정보', '电子': '전자', '系统': '시스템',
            '保护': '보호', '认证': '인증', '网络': '네트워크'
        }
        
        for chinese, korean in safe_replacements.items():
            text = text.replace(chinese, korean)
        
        # 문제 문자 제거
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]+', '', text)
        text = re.sub(r'[а-яё]+', '', text, flags=re.IGNORECASE)
        
        # 특수 기호 정리
        text = re.sub(r'[①②③④⑤➀➁❶❷❸❹❺]', '', text)
        text = re.sub(r'\bbo+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[bB][oO]+\b', '', text)
        
        # 불완전 한글 정리
        text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]{3,}(?![가-힣])', '', text)
        
        # 괄호 안 외국어 제거
        text = re.sub(r'\([^가-힣\s\d.,!?]*\)', '', text)
        
        # 연속 특수문자 정리
        problematic_fragments = [
            r'[^\w\s가-힣0-9.,!?()·\-\n""'']+',
            r'[A-Za-z]{10,}',
            r'\d{8,}'
        ]
        
        for pattern in problematic_fragments:
            text = re.sub(pattern, ' ', text)
        
        # 공백 및 구두점 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.,!?]{3,}', '.', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        text = text.strip()
        
        # 손상 검사
        if len(text) < original_length * 0.25 and original_length > 30:
            return self._attempt_text_recovery(original_text)
        
        return text
    
    def _attempt_text_recovery(self, corrupted_text: str) -> str:
        """손상된 텍스트 복구"""
        korean_parts = re.findall(r'[가-힣\s.,!?]+', corrupted_text)
        
        if korean_parts:
            recovered = ' '.join(korean_parts)
            recovered = re.sub(r'\s+', ' ', recovered).strip()
            
            if len(recovered) > 15:
                return recovered
        
        return ""
    
    def _evaluate_korean_quality_enhanced(self, text: str) -> float:
        if not text:
            return 0.0
        
        penalty_score = 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            penalty_score += 0.4
        
        if re.search(r'[ㄱ-ㅎㅏ-ㅣ]{3,}', text):
            penalty_score += 0.3
        
        if re.search(r'\bbo+\b', text, flags=re.IGNORECASE):
            penalty_score += 0.3
        
        if re.search(r'[①②③④⑤➀➁❶❷❸]', text):
            penalty_score += 0.2
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < 0.4:
            return max(0, korean_ratio * 0.4 - penalty_score)
        
        quality = korean_ratio * 0.85 - english_ratio * 0.1 - penalty_score
        
        # 길이 보정
        if 40 <= len(text) <= 350:
            quality += 0.15
        elif 25 <= len(text) <= 40:
            quality += 0.05
        elif len(text) < 25:
            quality -= 0.1
        
        # 전문 용어 보정
        professional_terms = ['법', '규정', '관리', '보안', '조치', '정책', '체계', '절차', '의무', '권리']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.06, 0.18)
        
        # 구조적 완성도
        if re.search(r'\.{3,}|,{3,}', text):
            quality -= 0.1
        
        sentence_count = len(re.findall(r'[.!?]', text))
        if sentence_count >= 2:
            quality += 0.05
        
        return max(0, min(1, quality))
    
    def _create_fallback_result(self, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            fallback_answer = str(random.randint(1, 5))
            return InferenceResult(
                response=fallback_answer,
                confidence=0.3,
                reasoning_quality=0.2,
                analysis_depth=1,
                korean_quality=1.0,
                fallback_used=True
            )
        else:
            fallback_answer = self._get_fallback_subjective_answer()
            return InferenceResult(
                response=fallback_answer,
                confidence=0.5,
                reasoning_quality=0.4,
                analysis_depth=1,
                korean_quality=0.85,
                fallback_used=True
            )
    
    def _get_fallback_subjective_answer(self) -> str:
        fallback_answers = [
            "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다.",
            "정보보안 정책과 절차를 수립하여 체계적인 보안 관리와 위험 평가를 수행해야 합니다.",
            "적절한 기술적, 관리적, 물리적 보안조치를 통해 정보자산을 안전하게 보호해야 합니다.",
            "법령에서 요구하는 안전성 확보조치를 이행하고 정기적인 점검을 통해 개선해야 합니다.",
            "위험관리 체계를 구축하여 예방적 관리와 사후 대응 방안을 마련해야 합니다."
        ]
        return random.choice(fallback_answers)
    
    def _evaluate_response(self, response: str, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            confidence = 0.6
            reasoning = 0.6
            
            if re.match(r'^[1-5]$', response.strip()):
                confidence = 0.9
                reasoning = 0.8
            elif re.search(r'[1-5]', response):
                confidence = 0.75
                reasoning = 0.65
            
            return InferenceResult(
                response=response,
                confidence=confidence,
                reasoning_quality=reasoning,
                analysis_depth=2,
                korean_quality=1.0
            )
        
        else:
            confidence = 0.7
            reasoning = 0.7
            
            length = len(response)
            if 60 <= length <= 350:
                confidence += 0.15
            elif 35 <= length <= 60:
                confidence += 0.05
            elif length > 350:
                confidence -= 0.05
            elif length < 35:
                confidence -= 0.15
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계', '절차', '의무', '권리']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 4:
                confidence += 0.15
                reasoning += 0.1
            elif keyword_count >= 2:
                confidence += 0.08
                reasoning += 0.05
            elif keyword_count >= 1:
                confidence += 0.03
            
            sentence_count = len(re.findall(r'[.!?]', response))
            if sentence_count >= 3:
                reasoning += 0.08
            elif sentence_count >= 2:
                reasoning += 0.05
            
            korean_quality = self._evaluate_korean_quality_enhanced(response)
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3,
                korean_quality=korean_quality
            )
    
    def _evaluate_generation_result(self, response: str, question_type: str, 
                                  strategy: GenerationStrategy, attempt: int) -> InferenceResult:
        """생성 결과 평가"""
        
        base_result = self._evaluate_response(response, question_type)
        
        # 전략별 보정
        if strategy.name == "precise":
            base_result.confidence *= 1.1
        elif strategy.name == "creative":
            base_result.reasoning_quality *= 1.1
        
        # 시도 횟수 페널티
        attempt_penalty = attempt * 0.05
        base_result.confidence = max(0.1, base_result.confidence - attempt_penalty)
        
        return base_result
    
    def _manage_cache(self):
        if len(self.response_cache) >= self.max_cache_size:
            keys_to_remove = list(self.response_cache.keys())[:self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.response_cache[key]
    
    def _perform_memory_cleanup(self):
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.memory_cleanup_counter % 40 == 0:
            if len(self.response_cache) > self.max_cache_size // 2:
                self._manage_cache()
    
    def _update_generation_stats(self, result: InferenceResult, success: bool):
        self.generation_stats["total_generations"] += 1
        if success:
            self.generation_stats["successful_generations"] += 1
        
        self.generation_stats["korean_quality_scores"].append(result.korean_quality)
        if len(self.generation_stats["korean_quality_scores"]) > 50:
            self.generation_stats["korean_quality_scores"] = self.generation_stats["korean_quality_scores"][-50:]
        
        total_time = self.generation_stats["avg_inference_time"] * (self.generation_stats["total_generations"] - 1)
        total_time += result.inference_time
        self.generation_stats["avg_inference_time"] = total_time / self.generation_stats["total_generations"]
        
        total_requests = self.generation_stats["total_generations"]
        self.generation_stats["cache_efficiency"] = self.cache_hits / max(total_requests, 1)
        
        # 성능 모니터링 업데이트
        self.performance_monitor["quality_trends"].append(result.korean_quality)
        self.performance_monitor["confidence_trends"].append(result.confidence)
        self.performance_monitor["inference_times"].append(result.inference_time)
        
        # 최근 50개 기록만 유지
        for key in ["quality_trends", "confidence_trends", "inference_times"]:
            if len(self.performance_monitor[key]) > 50:
                self.performance_monitor[key] = self.performance_monitor[key][-50:]
    
    def get_performance_stats(self) -> Dict:
        avg_korean_quality = 0.0
        if self.generation_stats["korean_quality_scores"]:
            avg_korean_quality = sum(self.generation_stats["korean_quality_scores"]) / len(self.generation_stats["korean_quality_scores"])
        
        success_rate = 0.0
        if self.generation_stats["total_generations"] > 0:
            success_rate = self.generation_stats["successful_generations"] / self.generation_stats["total_generations"]
        
        # 전략별 성능
        strategy_performance = {}
        for name, strategy in self.generation_strategies.items():
            strategy_performance[name] = {
                "success_rate": strategy.success_rate,
                "usage_count": strategy.usage_count
            }
        
        # 최근 트렌드
        recent_trends = {}
        for key, values in self.performance_monitor.items():
            if values and isinstance(values[0], (int, float)):
                recent_trends[f"avg_{key}"] = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        
        return {
            "model_name": self.model_name,
            "total_generations": self.generation_stats["total_generations"],
            "success_rate": success_rate,
            "avg_korean_quality": avg_korean_quality,
            "avg_inference_time": self.generation_stats["avg_inference_time"],
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "cache_efficiency": self.generation_stats["cache_efficiency"],
            "memory_cleanups": self.memory_cleanup_counter,
            "generation_failures": self.generation_stats["generation_failures"],
            "confidence_calibrations": self.generation_stats["confidence_calibrations"],
            "fallback_activations": self.generation_stats["fallback_activations"],
            "strategy_performance": strategy_performance,
            "recent_trends": recent_trends,
            "error_recovery_attempts": self.error_recovery["recovery_attempts"]
        }
    
    def cleanup(self):
        if self.verbose:
            stats = self.get_performance_stats()
            print(f"모델 통계: 생성 성공률 {stats['success_rate']:.1%}, 한국어 품질 {stats['avg_korean_quality']:.2f}")
            print(f"신뢰도 보정: {stats['confidence_calibrations']}회, 폴백 활성화: {stats['fallback_activations']}회")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        self.response_cache.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "generation_strategies": len(self.generation_strategies),
            "adaptive_parameters_enabled": self.adaptive_parameters["dynamic_adjustment"]
        }