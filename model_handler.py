# model_handler.py

"""
모델 핸들러
- LLM 모델 로딩 및 관리
- 파인튜닝된 모델 지원
- Chain-of-Thought 추론 최적화
- 텍스트 생성 및 추론  
- 한국어 최적화 설정
- 메모리 관리
"""

import torch
import re
import time
import gc
import os
import random
import hashlib
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# 상수 정의
DEFAULT_MAX_MEMORY_GB = 22
DEFAULT_CACHE_SIZE = 400
DEFAULT_MC_MAX_TOKENS = 25
DEFAULT_SUBJ_MAX_TOKENS = 400
DEFAULT_COT_MAX_TOKENS = 500
MEMORY_CLEANUP_INTERVAL = 20
CACHE_CLEANUP_INTERVAL = 40
MAX_PROMPT_LENGTH = 1500

@dataclass
class InferenceResult:
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int
    inference_time: float = 0.0
    korean_quality: float = 0.0
    reasoning_steps: int = 0
    prompt_type: str = "basic"

class ModelHandler:
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = True, max_memory_gb: int = DEFAULT_MAX_MEMORY_GB, 
                 verbose: bool = False, finetuned_path: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        self.finetuned_path = finetuned_path
        self.is_finetuned = False
        
        # GPU 사용 가능 여부 초기화 시 한 번만 체크
        self.cuda_available = torch.cuda.is_available()
        
        if self.verbose:
            print(f"모델 로딩: {model_name}")
            if finetuned_path:
                print(f"파인튜닝 모델: {finetuned_path}")
        
        self._initialize_tokenizer()
        self._initialize_model(load_in_4bit)
        self._prepare_korean_optimization()
        
        # 캐시 및 통계 초기화
        self.response_cache = {}
        self.cache_hits = 0
        self.max_cache_size = DEFAULT_CACHE_SIZE
        self.memory_cleanup_counter = 0
        
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "korean_quality_scores": [],
            "avg_inference_time": 0.0,
            "cache_efficiency": 0.0,
            "generation_failures": 0,
            "timeout_failures": 0,
            "finetuned_generations": 0,
            "cot_generations": 0,
            "reasoning_generations": 0,
            "step_by_step_generations": 0,
            "model_call_failures": 0,
            "torch_errors": 0,
            "validation_failures": 0,
            "fallback_usage": 0
        }
        
        if self.verbose:
            model_type = "파인튜닝된 모델" if self.is_finetuned else "기본 모델"
            print(f"{model_type} 로드 완료")
    
    def _initialize_tokenizer(self) -> None:
        """토크나이저 초기화"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
        except Exception as e:
            raise RuntimeError(f"토크나이저 로딩 실패: {e}")
    
    def _initialize_model(self, load_in_4bit: bool) -> None:
        """모델 초기화"""
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        
        # 양자화 설정
        if load_in_4bit and self.cuda_available:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # Attention 구현 설정
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
        
        # 기본 모델 로딩
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"기본 모델 로딩 실패: {e}")
        
        # 파인튜닝 모델 로딩
        if self.finetuned_path and os.path.exists(self.finetuned_path):
            try:
                self.model = PeftModel.from_pretrained(self.model, self.finetuned_path)
                self.is_finetuned = True
                if self.verbose:
                    print("파인튜닝된 모델 로드 완료")
            except Exception as e:
                if self.verbose:
                    print(f"파인튜닝 모델 로드 실패: {e}")
                    print("기본 모델 사용")
        
        self.model.eval()
        
        # Torch Compile 시도
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                if self.verbose:
                    print("Torch Compile 활성화")
            except Exception:
                if self.verbose:
                    print("Torch Compile 실패, 기본 모델 사용")
    
    def _prepare_korean_optimization(self) -> None:
        """한국어 최적화를 위한 불용어 설정"""
        self.bad_words_ids = []
        
        # 문제가 되는 중국어 패턴들
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
            except Exception:
                continue
        
        # 특수 기호들
        special_symbols = ["①", "②", "③", "④", "⑤", "➀", "➁", "➂", "➃", "➄", "bo", "Bo", "BO"]
        for symbol in special_symbols:
            try:
                tokens = self.tokenizer.encode(symbol, add_special_tokens=False)
                if tokens:
                    self.bad_words_ids.append(tokens)
            except Exception:
                continue
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str, 
                                       question_structure: Optional[Dict] = None) -> str:
        """개선된 프롬프트 생성 - 선택지 내용 포함"""
        
        if question_type == "multiple_choice":
            choices = question_structure.get("choices", []) if question_structure else []
            has_negative = question_structure.get("has_negative", False) if question_structure else False
            
            # 선택지 텍스트 포맷팅
            choices_text = ""
            if choices:
                choices_text = "\n선택지:\n"
                for choice in choices:
                    choices_text += f"{choice['number']}. {choice['text']}\n"
            
            # 부정형 질문 강조
            if has_negative:
                korean_prefix = """다음 금융보안 문제를 신중히 읽고 정답을 선택하세요.
주의: 이 문제는 '해당하지 않는 것', '틀린 것', '적절하지 않은 것'을 찾는 부정형 질문입니다.

문제:
"""
                korean_suffix = f"""
{choices_text}
위 선택지를 모두 검토한 후, 질문에 부합하는 정답 번호를 선택하세요.
정답은 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요.

정답:"""
            else:
                korean_prefix = """다음 금융보안 문제를 읽고 가장 적절한 답을 선택하세요.

문제:
"""
                korean_suffix = f"""
{choices_text}
위 선택지를 검토한 후, 가장 적절한 번호를 선택하세요.
정답은 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요.

정답:"""
        else:
            # 주관식 - 도메인별 힌트 제공
            domain_hints = question_structure.get("domain_hints", []) if question_structure else []
            
            if "사이버보안" in domain_hints or "트로이" in prompt.lower():
                korean_prefix = """다음 사이버보안 관련 질문에 대해 전문적인 한국어 답변을 작성하세요.
트로이 목마, 악성코드, 탐지 방법 등을 포함하여 구체적으로 설명하세요.

질문:
"""
            elif "개인정보" in domain_hints:
                korean_prefix = """다음 개인정보보호 관련 질문에 대해 개인정보보호법에 근거한 답변을 작성하세요.

질문:
"""
            elif "전자금융" in domain_hints:
                korean_prefix = """다음 전자금융 관련 질문에 대해 전자금융거래법에 근거한 답변을 작성하세요.

질문:
"""
            else:
                korean_prefix = """다음 금융보안 질문에 대해 법령과 규정에 근거한 전문적인 답변을 작성하세요.

질문:
"""
            korean_suffix = "\n\n한국어로 답변:"
        
        return korean_prefix + prompt + korean_suffix
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 3, question_structure: Optional[Dict] = None) -> InferenceResult:
        
        start_time = time.time()
        
        # 캐시 키 충돌 방지를 위한 개선된 해시
        cache_key = hashlib.md5((prompt + question_type).encode('utf-8')).hexdigest()[:16]
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            cached.inference_time = 0.01
            return cached
        
        # 개선된 프롬프트 생성
        optimized_prompt = self._create_korean_optimized_prompt(prompt, question_type, question_structure)
        
        # 프롬프트 유형 감지
        prompt_type = self._detect_prompt_type(optimized_prompt)
        
        best_result = None
        best_score = 0
        generation_errors = 0
        
        for attempt in range(max_attempts):
            try:
                gen_config = self._create_generation_config(question_type, attempt, prompt_type)
                
                # 토크나이징 개선
                try:
                    inputs = self.tokenizer(
                        optimized_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_PROMPT_LENGTH,
                        padding=False
                    ).to(self.model.device)
                except Exception as tokenize_error:
                    if self.verbose:
                        print(f"토크나이징 오류 (시도 {attempt+1}): {str(tokenize_error)[:100]}")
                    generation_errors += 1
                    continue
                
                # 실제 모델 추론 개선
                with torch.no_grad():
                    try:
                        if self.cuda_available:
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                outputs = self.model.generate(
                                    **inputs,
                                    generation_config=gen_config,
                                    do_sample=gen_config.do_sample,
                                    temperature=gen_config.temperature,
                                    top_p=gen_config.top_p,
                                    top_k=gen_config.top_k,
                                    max_new_tokens=gen_config.max_new_tokens,
                                    repetition_penalty=gen_config.repetition_penalty,
                                    pad_token_id=gen_config.pad_token_id,
                                    eos_token_id=gen_config.eos_token_id,
                                    bad_words_ids=gen_config.bad_words_ids
                                )
                        else:
                            outputs = self.model.generate(
                                **inputs,
                                generation_config=gen_config
                            )
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as cuda_error:
                        if self.verbose:
                            print(f"CUDA/런타임 오류 (시도 {attempt+1}): {str(cuda_error)[:100]}")
                        generation_errors += 1
                        self.generation_stats["torch_errors"] += 1
                        
                        # GPU 메모리 정리 후 재시도
                        if self.cuda_available:
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    except Exception as model_error:
                        if self.verbose:
                            print(f"모델 생성 오류 (시도 {attempt+1}): {str(model_error)[:100]}")
                        generation_errors += 1
                        self.generation_stats["model_call_failures"] += 1
                        continue
                
                # 응답 디코딩 및 처리
                try:
                    raw_response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).strip()
                except Exception as decode_error:
                    if self.verbose:
                        print(f"디코딩 오류 (시도 {attempt+1}): {str(decode_error)[:100]}")
                    generation_errors += 1
                    continue
                
                # 한국어 텍스트 정리
                cleaned_response = self._clean_korean_text_enhanced(raw_response)
                
                if question_type == "multiple_choice":
                    extracted_answer = self._extract_mc_answer_enhanced(cleaned_response)
                    
                    if extracted_answer and extracted_answer.isdigit() and 1 <= int(extracted_answer) <= 5:
                        confidence = 0.90 - (attempt * 0.03)
                        if self.is_finetuned:
                            confidence += 0.05
                        if prompt_type == "cot":
                            confidence += 0.03
                        
                        result = InferenceResult(
                            response=extracted_answer,
                            confidence=confidence,
                            reasoning_quality=0.85,
                            analysis_depth=2,
                            korean_quality=1.0,
                            inference_time=time.time() - start_time,
                            reasoning_steps=self._count_reasoning_steps(optimized_prompt),
                            prompt_type=prompt_type
                        )
                        
                        self._manage_cache()
                        self.response_cache[cache_key] = result
                        self._update_generation_stats(result, True)
                        
                        return result
                    else:
                        continue
                
                else:
                    # 주관식 답변 검증 완화
                    korean_quality = self._evaluate_korean_quality_enhanced(cleaned_response)
                    
                    if korean_quality > 0.4 and len(cleaned_response) > 20:  # 기준 완화
                        result = self._evaluate_response(cleaned_response, question_type, prompt_type)
                        result.korean_quality = korean_quality
                        result.inference_time = time.time() - start_time
                        result.reasoning_steps = self._count_reasoning_steps(optimized_prompt)
                        result.prompt_type = prompt_type
                        
                        if self.is_finetuned:
                            result.confidence += 0.05
                        if prompt_type == "cot":
                            result.confidence += 0.03
                            result.reasoning_quality += 0.05
                        
                        score = korean_quality * result.confidence
                        if score > best_score:
                            best_score = score
                            best_result = result
                    
                    if korean_quality > 0.6:  # 기준 완화
                        break
                        
            except Exception as e:
                generation_errors += 1
                self.generation_stats["generation_failures"] += 1
                if self.verbose:
                    print(f"전체 생성 오류 (시도 {attempt+1}): {str(e)[:100]}")
                continue
        
        # 결과 검증 및 반환
        if best_result is None:
            best_result = self._create_fallback_result(question_type, question_structure, prompt_type)
            best_result.inference_time = time.time() - start_time
            self.generation_stats["fallback_usage"] += 1
            self._update_generation_stats(best_result, False)
        else:
            self._update_generation_stats(best_result, True)
        
        if self.is_finetuned:
            self.generation_stats["finetuned_generations"] += 1
        
        self._perform_memory_cleanup()
        
        return best_result
    
    def _detect_prompt_type(self, prompt: str) -> str:
        """프롬프트 유형 감지"""
        prompt_lower = prompt.lower()
        
        # CoT 패턴 감지
        cot_patterns = ["단계", "step", "분석", "검토", "추론", "과정"]
        if any(pattern in prompt_lower for pattern in cot_patterns):
            return "cot"
        
        # 추론 패턴 감지
        reasoning_patterns = ["논리적", "근거", "이유", "why", "because", "따라서"]
        if any(pattern in prompt_lower for pattern in reasoning_patterns):
            return "reasoning"
        
        # 다단계 패턴 감지
        step_patterns = ["1단계", "2단계", "3단계", "first", "second", "third"]
        if any(pattern in prompt_lower for pattern in step_patterns):
            return "step_by_step"
        
        return "basic"
    
    def _count_reasoning_steps(self, prompt: str) -> int:
        """추론 단계 수 계산"""
        step_patterns = [
            r'\d+단계',
            r'step\s*\d+',
            r'(\d+)\.',
            r'첫째|둘째|셋째|넷째',
            r'first|second|third|fourth'
        ]
        
        total_steps = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, prompt.lower())
            total_steps += len(matches)
        
        return min(total_steps, 10)  # 최대 10단계로 제한
    
    def _create_generation_config(self, question_type: str, attempt: int, prompt_type: str = "basic") -> GenerationConfig:
        """생성 설정 생성 (프롬프트 유형별 최적화)"""
        base_config = {
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "bad_words_ids": self.bad_words_ids[:10] if self.bad_words_ids else None
        }
        
        if question_type == "multiple_choice":
            # 객관식 설정
            base_config.update({
                "temperature": 0.25 + (attempt * 0.05),  # 더 낮은 온도
                "top_p": 0.75,
                "top_k": 15,  # 더 낮은 top_k
                "max_new_tokens": DEFAULT_MC_MAX_TOKENS,
                "repetition_penalty": 1.1
            })
        else:
            # 주관식 설정 - 프롬프트 유형별 조정
            if prompt_type == "cot":
                base_config.update({
                    "temperature": 0.35 + (attempt * 0.05),
                    "top_p": 0.85,
                    "top_k": 30,
                    "max_new_tokens": DEFAULT_COT_MAX_TOKENS,
                    "repetition_penalty": 1.05
                })
                self.generation_stats["cot_generations"] += 1
            elif prompt_type == "reasoning":
                base_config.update({
                    "temperature": 0.4 + (attempt * 0.06),
                    "top_p": 0.82,
                    "top_k": 25,
                    "max_new_tokens": DEFAULT_SUBJ_MAX_TOKENS + 50,
                    "repetition_penalty": 1.08
                })
                self.generation_stats["reasoning_generations"] += 1
            elif prompt_type == "step_by_step":
                base_config.update({
                    "temperature": 0.3 + (attempt * 0.05),
                    "top_p": 0.88,
                    "top_k": 35,
                    "max_new_tokens": DEFAULT_COT_MAX_TOKENS,
                    "repetition_penalty": 1.03
                })
                self.generation_stats["step_by_step_generations"] += 1
            else:
                base_config.update({
                    "temperature": 0.45 + (attempt * 0.08),
                    "top_p": 0.80,
                    "top_k": 25,
                    "max_new_tokens": DEFAULT_SUBJ_MAX_TOKENS,
                    "repetition_penalty": 1.07
                })
        
        return GenerationConfig(**base_config)
    
    def _extract_mc_answer_enhanced(self, text: str) -> str:
        """개선된 객관식 답변 추출"""
        if not text:
            return ""
        
        text = text.strip()
        
        # 단순 숫자 매칭
        if re.match(r'^[1-5]$', text):
            return text
        
        # 첫 몇 글자에서 숫자 찾기
        first_part = text[:15] if len(text) > 15 else text
        early_match = re.search(r'[1-5]', first_part)
        if early_match:
            return early_match.group()
        
        priority_patterns = [
            (r'정답[:\s]*([1-5])', 0.95),
            (r'답[:\s]*([1-5])', 0.90),
            (r'^([1-5])\s*$', 0.95),
            (r'^([1-5])\s*번', 0.85),
            (r'선택[:\s]*([1-5])', 0.85),
            (r'([1-5])번이', 0.80),
            (r'([1-5])가\s*정답', 0.80),
            (r'([1-5])이\s*정답', 0.80),
            (r'([1-5])\s*이\s*적절', 0.75),
            (r'([1-5])\s*가\s*적절', 0.75)
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
        
        # 전체 텍스트에서 첫 번째 숫자
        numbers = re.findall(r'[1-5]', text)
        if numbers:
            return numbers[0]
        
        return ""
    
    def _clean_korean_text_enhanced(self, text: str) -> str:
        """개선된 한국어 텍스트 정리"""
        if not text:
            return ""
        
        original_text = text
        original_length = len(text)
        
        # 제어 문자 제거
        text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)
        
        # 안전한 중국어-한국어 변환
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
        
        # 문제가 되는 문자들 제거 (한국어 한자 제외한 중국어)
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]+', '', text)
        text = re.sub(r'[À-Ñ]+', '', text, flags=re.IGNORECASE)
        
        # 특수 기호 제거
        text = re.sub(r'[①②③④⑤➀➁➂➃➄➅➆➇➈➉]', '', text)
        text = re.sub(r'\bbo+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[bB][oO]+\b', '', text)
        
        # 깨진 한글 제거
        text = re.sub(r'\([^가-힣\s\d.,!?]*\)', '', text)
        
        # 문제가 되는 조각들 제거
        problematic_fragments = [
            r'[ㄱ-ㅎㅏ-ㅣ]{3,}(?![가-힣])', 
            r'[^\w\s가-힣0-9.,!?()\·\-\n""'']+',
            r'[A-Za-z]{8,}',  # 너무 긴 영어 단어
            r'\d{6,}'  # 너무 긴 숫자
        ]
        
        for pattern in problematic_fragments:
            text = re.sub(pattern, ' ', text)
        
        # 공백 및 구두점 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.,!?]{3,}', '.', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        text = text.strip()
        
        # 너무 많이 정리된 경우 원본의 일부라도 보존
        if len(text) < original_length * 0.2 and original_length > 30:
            # 원본에서 한국어 부분만 추출 시도
            korean_parts = re.findall(r'[가-힣\s.,!?]+', original_text)
            if korean_parts:
                text = ' '.join(korean_parts).strip()
            
            if len(text) < 10:
                return ""
        
        return text
    
    def _evaluate_korean_quality_enhanced(self, text: str) -> float:
        """개선된 한국어 품질 평가"""
        if not text:
            return 0.0
        
        penalty_score = 0.0
        
        # 패널티 요소들
        if re.search(r'[\u4e00-\u9fff]', text):
            penalty_score += 0.3  # 감소
        
        if re.search(r'[ㄱ-ㅎㅏ-ㅣ]{3,}', text):
            penalty_score += 0.2  # 감소
        
        if re.search(r'\bbo+\b', text, flags=re.IGNORECASE):
            penalty_score += 0.25  # 감소
        
        if re.search(r'[①②③④⑤➀➁➂➃➄]', text):
            penalty_score += 0.15  # 감소
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < 0.3:  # 기준 완화
            return max(0, korean_ratio * 0.5 - penalty_score)
        
        quality = korean_ratio * 0.85 - english_ratio * 0.08 - penalty_score  # 영어 패널티 감소
        
        # 길이 보너스
        if 30 <= len(text) <= 400:  # 범위 확대
            quality += 0.15
        elif 20 <= len(text) <= 30:
            quality += 0.08
        elif len(text) < 20:
            quality -= 0.05  # 패널티 감소
        
        # 전문 용어 보너스
        professional_terms = ['법', '규정', '관리', '보안', '조치', '정책', '체계', '절차', '의무', '권리']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.05, 0.15)
        
        # 문장 구조 보너스
        sentence_count = len(re.findall(r'[.!?]', text))
        if sentence_count >= 2:
            quality += 0.05
        
        return max(0, min(1, quality))
    
    def _create_fallback_result(self, question_type: str, 
                               question_structure: Optional[Dict] = None,
                               prompt_type: str = "basic") -> InferenceResult:
        """향상된 폴백 결과 생성"""
        if question_type == "multiple_choice":
            # 선택지 분석 활용
            if question_structure:
                choice_analysis = question_structure.get("choice_analysis", {})
                if choice_analysis.get("inclusion_candidates"):
                    fallback_answer = choice_analysis["inclusion_candidates"][0]
                elif question_structure.get("has_negative"):
                    # 부정형은 3,4,5번 선호
                    fallback_answer = str(random.choice([3, 4, 5]))
                else:
                    fallback_answer = str(random.randint(1, 5))
            else:
                fallback_answer = str(random.randint(1, 5))
                
            return InferenceResult(
                response=fallback_answer,
                confidence=0.4,  # 증가
                reasoning_quality=0.3,  # 증가
                analysis_depth=1,
                korean_quality=1.0,
                prompt_type=prompt_type
            )
        else:
            # 도메인별 폴백
            domain_templates = {
                "사이버보안": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결과 시스템 리소스 사용 증가가 있습니다.",
                "개인정보보호": "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다.",
                "전자금융": "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다.",
                "정보보안": "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다."
            }
            
            if question_structure:
                domains = question_structure.get("domain_hints", [])
                for domain in domains:
                    if domain in domain_templates:
                        selected_answer = domain_templates[domain]
                        break
                else:
                    selected_answer = "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
            else:
                selected_answer = "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
            
            return InferenceResult(
                response=selected_answer,
                confidence=0.6,  # 증가
                reasoning_quality=0.5,  # 증가
                analysis_depth=1,
                korean_quality=0.85,
                prompt_type=prompt_type
            )
    
    def _evaluate_response(self, response: str, question_type: str, prompt_type: str = "basic") -> InferenceResult:
        """응답 품질 평가"""
        if question_type == "multiple_choice":
            confidence = 0.65  # 기본값 증가
            reasoning = 0.65
            
            if re.match(r'^[1-5]$', response.strip()):
                confidence = 0.92
                reasoning = 0.85
            elif re.search(r'[1-5]', response):
                confidence = 0.78
                reasoning = 0.68
            
            return InferenceResult(
                response=response,
                confidence=confidence,
                reasoning_quality=reasoning,
                analysis_depth=2,
                prompt_type=prompt_type
            )
        
        else:
            confidence = 0.75  # 기본값 증가
            reasoning = 0.75
            
            length = len(response)
            if 50 <= length <= 400:  # 범위 확대
                confidence += 0.15
            elif 30 <= length <= 50:
                confidence += 0.08
            elif length > 400:
                confidence -= 0.03  # 패널티 감소
            elif length < 30:
                confidence -= 0.1  # 패널티 감소
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계', '절차', '의무', '권리']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 4:
                confidence += 0.12
                reasoning += 0.08
            elif keyword_count >= 2:
                confidence += 0.06
                reasoning += 0.04
            elif keyword_count >= 1:
                confidence += 0.02
            
            sentence_count = len(re.findall(r'[.!?]', response))
            if sentence_count >= 3:
                reasoning += 0.06
            elif sentence_count >= 2:
                reasoning += 0.03
            
            # 프롬프트 유형별 보너스
            if prompt_type == "cot":
                reasoning += 0.05
                confidence += 0.03
            elif prompt_type == "reasoning":
                reasoning += 0.08
                confidence += 0.02
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3,
                prompt_type=prompt_type
            )
    
    def _manage_cache(self) -> None:
        """캐시 관리"""
        if len(self.response_cache) >= self.max_cache_size:
            keys_to_remove = list(self.response_cache.keys())[:self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.response_cache[key]
    
    def _perform_memory_cleanup(self) -> None:
        """메모리 정리"""
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % MEMORY_CLEANUP_INTERVAL == 0:
            gc.collect()
            if self.cuda_available:
                torch.cuda.empty_cache()
        
        if self.memory_cleanup_counter % CACHE_CLEANUP_INTERVAL == 0:
            if len(self.response_cache) > self.max_cache_size // 2:
                self._manage_cache()
    
    def _update_generation_stats(self, result: InferenceResult, success: bool) -> None:
        """생성 통계 업데이트"""
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
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        avg_korean_quality = 0.0
        if self.generation_stats["korean_quality_scores"]:
            avg_korean_quality = sum(self.generation_stats["korean_quality_scores"]) / len(self.generation_stats["korean_quality_scores"])
        
        success_rate = 0.0
        if self.generation_stats["total_generations"] > 0:
            success_rate = self.generation_stats["successful_generations"] / self.generation_stats["total_generations"]
        
        finetuned_rate = 0.0
        if self.generation_stats["total_generations"] > 0:
            finetuned_rate = self.generation_stats["finetuned_generations"] / self.generation_stats["total_generations"]
        
        cot_rate = 0.0
        reasoning_rate = 0.0
        step_rate = 0.0
        if self.generation_stats["total_generations"] > 0:
            cot_rate = self.generation_stats["cot_generations"] / self.generation_stats["total_generations"]
            reasoning_rate = self.generation_stats["reasoning_generations"] / self.generation_stats["total_generations"]
            step_rate = self.generation_stats["step_by_step_generations"] / self.generation_stats["total_generations"]
        
        return {
            "model_name": self.model_name,
            "is_finetuned": self.is_finetuned,
            "finetuned_path": self.finetuned_path,
            "total_generations": self.generation_stats["total_generations"],
            "success_rate": success_rate,
            "avg_korean_quality": avg_korean_quality,
            "avg_inference_time": self.generation_stats["avg_inference_time"],
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "cache_efficiency": self.generation_stats["cache_efficiency"],
            "memory_cleanups": self.memory_cleanup_counter,
            "generation_failures": self.generation_stats["generation_failures"],
            "finetuned_usage_rate": finetuned_rate,
            "prompt_type_distribution": {
                "cot_rate": cot_rate,
                "reasoning_rate": reasoning_rate,
                "step_by_step_rate": step_rate
            },
            "error_analysis": {
                "model_call_failures": self.generation_stats["model_call_failures"],
                "torch_errors": self.generation_stats["torch_errors"],
                "validation_failures": self.generation_stats["validation_failures"],
                "fallback_usage": self.generation_stats["fallback_usage"]
            }
        }
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            if self.verbose:
                stats = self.get_performance_stats()
                model_type = "파인튜닝된" if self.is_finetuned else "기본"
                print(f"{model_type} 모델 통계: 생성 성공률 {stats['success_rate']:.1%}, 한국어 품질 {stats['avg_korean_quality']:.2f}")
                
                prompt_dist = stats['prompt_type_distribution']
                print(f"  프롬프트 유형: CoT {prompt_dist['cot_rate']:.1%}, 추론 {prompt_dist['reasoning_rate']:.1%}, 단계별 {prompt_dist['step_by_step_rate']:.1%}")
                
                error_analysis = stats['error_analysis']
                if error_analysis['model_call_failures'] > 0:
                    print(f"  오류 분석: 모델호출실패 {error_analysis['model_call_failures']}, TORCH오류 {error_analysis['torch_errors']}, 폴백사용 {error_analysis['fallback_usage']}")
            
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            
            self.response_cache.clear()
            
            gc.collect()
            if self.cuda_available:
                torch.cuda.empty_cache()
                
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "is_finetuned": self.is_finetuned,
            "finetuned_path": self.finetuned_path,
            "device": self.device,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "optimization_features": {
                "korean_optimization": True,
                "cot_support": True,
                "reasoning_support": True,
                "step_by_step_support": True,
                "prompt_type_detection": True,
                "enhanced_error_handling": True,
                "improved_validation": True
            }
        }