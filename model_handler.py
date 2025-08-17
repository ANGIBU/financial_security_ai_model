# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 학습 데이터 저장
- 질문 의도 기반 답변 생성
- LLM 기반 텍스트 생성 준수
"""

import torch
import re
import time
import gc
import random
import pickle
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 설정 파일 import
from config import (
    DEFAULT_MODEL_NAME, MODEL_CONFIG, GENERATION_CONFIG, 
    OPTIMIZATION_CONFIG, PKL_DIR, JSON_CONFIG_FILES,
    MEMORY_CONFIG, get_device, TEXT_CLEANUP_CONFIG,
    KOREAN_TYPO_MAPPING, TOKENIZER_SAFETY_CONFIG, check_text_safety
)

class SimpleModelHandler:
    """모델 핸들러 - LLM 생성 중심 버전"""
    
    def __init__(self, model_name: str = None, verbose: bool = False):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.verbose = verbose
        self.device = get_device()
        
        # pkl 저장 폴더 생성
        self.pkl_dir = PKL_DIR
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 텍스트 정리 설정 로드
        self.text_cleanup_config = TEXT_CLEANUP_CONFIG
        self.korean_typo_mapping = KOREAN_TYPO_MAPPING
        
        # JSON 설정 파일에서 데이터 로드
        self._load_json_configs()
        
        # 성능 최적화 설정
        self.optimization_config = OPTIMIZATION_CONFIG
        
        # 학습 데이터 저장
        self.learning_data = self.learning_data_structure.copy()
        
        # 이전 학습 데이터 로드
        self._load_learning_data()
        
        if verbose:
            print(f"모델 로딩: {self.model_name}")
            print(f"디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=MODEL_CONFIG['trust_remote_code'],
            use_fast=MODEL_CONFIG['use_fast_tokenizer'],
            clean_up_tokenization_spaces=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG['torch_dtype']),
            device_map=MODEL_CONFIG['device_map'],
            trust_remote_code=MODEL_CONFIG['trust_remote_code'],
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # 워밍업
        self._warmup()
        
        if verbose:
            print("모델 로딩 완료")
        
        # 학습 데이터 로드 현황
        if len(self.learning_data["successful_answers"]) > 0 and verbose:
            print(f"이전 학습 데이터 로드: 성공 {len(self.learning_data['successful_answers'])}개, 실패 {len(self.learning_data['failed_answers'])}개")
    
    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            # model_config.json 로드
            with open(JSON_CONFIG_FILES['model_config'], 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            # 모델 관련 데이터 할당
            self.mc_context_patterns = model_config['mc_context_patterns']
            self.intent_specific_prompts = model_config['intent_specific_prompts']
            self.answer_distributions = model_config['answer_distribution_default'].copy()
            self.mc_answer_counts = model_config['mc_answer_counts_default'].copy()
            self.learning_data_structure = model_config['learning_data_structure']
            
            print("모델 설정 파일 로드 완료")
            
        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()
    
    def _load_default_configs(self):
        """기본 설정 로드 (JSON 파일 로드 실패 시)"""
        print("기본 설정으로 대체합니다.")
        
        # 최소한의 기본 설정
        self.mc_context_patterns = {
            "negative_keywords": ["해당하지.*않는", "적절하지.*않는", "옳지.*않는"],
            "positive_keywords": ["맞는.*것", "옳은.*것", "적절한.*것"],
            "domain_specific_patterns": {}
        }
        
        self.intent_specific_prompts = {
            "기관_묻기": ["질문에서 요구하는 구체적인 기관명을 정확히 답변하세요."],
            "특징_묻기": ["해당 항목의 핵심적인 특징들을 구체적으로 설명하세요."],
            "지표_묻기": ["탐지 지표와 징후를 중심으로 구체적으로 설명하세요."]
        }
        
        self.answer_distributions = {
            3: {"1": 0, "2": 0, "3": 0},
            4: {"1": 0, "2": 0, "3": 0, "4": 0},
            5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        }
        
        self.mc_answer_counts = {3: 0, 4: 0, 5: 0}
        
        self.learning_data_structure = {
            "successful_answers": [],
            "failed_answers": [],
            "question_patterns": {},
            "answer_quality_scores": []
        }
    
    def _load_learning_data(self):
        """이전 학습 데이터 로드"""
        learning_file = self.pkl_dir / "learning_data.pkl"
        
        if learning_file.exists():
            try:
                with open(learning_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.learning_data.update(saved_data)
                if self.verbose:
                    print("학습 데이터 로드 완료")
            except Exception as e:
                if self.verbose:
                    print(f"학습 데이터 로드 오류: {e}")
    
    def _save_learning_data(self):
        """학습 데이터 저장"""
        learning_file = self.pkl_dir / "learning_data.pkl"
        
        try:
            # 저장할 데이터 정리 (최근 데이터만)
            save_data = {
                "successful_answers": self.learning_data["successful_answers"][-MEMORY_CONFIG['max_learning_records']['successful_answers']:],
                "failed_answers": self.learning_data["failed_answers"][-MEMORY_CONFIG['max_learning_records']['failed_answers']:],
                "question_patterns": self.learning_data["question_patterns"],
                "answer_quality_scores": self.learning_data["answer_quality_scores"][-MEMORY_CONFIG['max_learning_records']['quality_scores']:],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
    def generate_mc_answer_with_hints(self, question: str, max_choice: int, domain: str, 
                                     context_hint: Dict = None, pattern_hint: Dict = None) -> str:
        """힌트를 활용한 객관식 답변 생성 - LLM 생성 중심"""
        
        # 힌트 기반 프롬프트 구성
        prompt_parts = ["다음 객관식 문제를 분석하여 정답을 선택하세요.\n"]
        
        # 컨텍스트 힌트 추가
        if context_hint and context_hint.get("is_negative"):
            prompt_parts.append("이 문제는 해당하지 않거나 적절하지 않은 것을 찾는 문제입니다.")
        elif context_hint and context_hint.get("is_positive"):
            prompt_parts.append("이 문제는 가장 적절하거나 올바른 것을 찾는 문제입니다.")
        
        # 패턴 힌트 추가
        if pattern_hint and pattern_hint.get("guidance"):
            prompt_parts.append(f"힌트: {pattern_hint['guidance']}")
        
        # 도메인별 추가 가이드
        domain_guides = {
            "금융투자": "자본시장법상 금융투자업의 구분을 고려하여 분석하세요.",
            "위험관리": "위험관리 계획 수립 시 고려해야 할 요소들을 확인하세요.",
            "개인정보보호": "개인정보보호 관리체계에서 중요한 요소를 판단하세요.",
            "전자금융": "전자금융거래법에 따른 관련 기관과 역할을 고려하세요.",
            "사이버보안": "사이버보안 위협의 특성과 대응 방안을 고려하세요."
        }
        
        if domain in domain_guides:
            prompt_parts.append(domain_guides[domain])
        
        prompt_parts.append(f"\n{question}")
        prompt_parts.append(f"\n위 문제를 신중히 분석하고, 1부터 {max_choice}번 중 하나의 정답 번호만 답하세요.")
        prompt_parts.append("\n정답:")
        
        prompt = "\n".join(prompt_parts)
        
        # LLM으로 생성
        answer = self._generate_with_llm(prompt, "multiple_choice", max_choice)
        
        # 후처리
        return self._process_mc_answer(answer, question, max_choice, domain)
    
    def generate_mc_answer_retry(self, question: str, max_choice: int, domain: str) -> str:
        """객관식 답변 재시도 생성"""
        
        # 간단한 재시도 프롬프트
        prompt = f"""문제를 다시 분석하여 정답을 선택하세요.

{question}

1부터 {max_choice}번 중에서 정답 번호 하나만 선택하세요.

정답:"""
        
        # 더 보수적인 설정으로 재생성
        answer = self._generate_with_llm_conservative(prompt, max_choice)
        
        return self._process_mc_answer(answer, question, max_choice, domain)
    
    def generate_subj_answer_with_knowledge(self, question: str, domain: str, intent_analysis: Dict, 
                                          knowledge_hints: Dict, institution_hints: Dict = None) -> str:
        """지식 힌트를 활용한 주관식 답변 생성 - LLM 생성 중심"""
        
        # 힌트 기반 프롬프트 구성
        prompt_parts = [f"다음 {domain} 분야 질문에 대해 한국어로만 정확한 답변을 작성하세요.\n"]
        
        # 구조 가이드 추가
        if knowledge_hints.get("structure_guidance"):
            prompt_parts.append(f"답변 구조: {knowledge_hints['structure_guidance']}")
        
        # 내용 방향 추가
        if knowledge_hints.get("content_direction"):
            prompt_parts.append(f"포함할 내용: {knowledge_hints['content_direction']}")
        
        # 핵심 개념 힌트
        if knowledge_hints.get("key_concepts"):
            concepts = ", ".join(knowledge_hints["key_concepts"])
            prompt_parts.append(f"핵심 개념: {concepts}")
        
        # 기관 정보 힌트 (직접 답변 아님)
        if institution_hints:
            if institution_hints.get("institution_name"):
                prompt_parts.append(f"관련 기관 정보: {institution_hints['institution_name']}")
            if institution_hints.get("role"):
                prompt_parts.append(f"기관 역할: {institution_hints['role']}")
        
        # 의도별 특별 가이드
        intent_type = intent_analysis.get("primary_intent", "일반")
        intent_guides = {
            "기관_묻기": "구체적인 기관명과 그 역할을 명확히 제시하세요.",
            "특징_묻기": "주요 특징과 특성을 체계적으로 설명하세요.",
            "지표_묻기": "탐지 가능한 지표와 징후를 구체적으로 나열하세요.",
            "방안_묻기": "실행 가능한 대응 방안과 절차를 제시하세요.",
            "절차_묻기": "단계별 처리 절차를 순서대로 설명하세요.",
            "조치_묻기": "필요한 보안조치와 관리조치를 설명하세요."
        }
        
        if intent_type in intent_guides:
            prompt_parts.append(intent_guides[intent_type])
        
        prompt_parts.append(f"\n질문: {question}")
        prompt_parts.append("\n반드시 한국어로만 작성하고, 명확하고 정확한 전문 용어를 사용하여 답변하세요.")
        prompt_parts.append("\n답변:")
        
        prompt = "\n".join(prompt_parts)
        
        # LLM으로 생성
        answer = self._generate_with_llm(prompt, "subjective", 5)
        
        return answer
    
    def regenerate_answer_safe(self, question: str, domain: str, intent_analysis: Dict) -> str:
        """안전한 답변 재생성"""
        
        # 단순화된 프롬프트로 재생성
        prompt = f"""다음 질문에 대해 한국어로만 간단명료하게 답변하세요.

질문: {question}

전문적이고 정확한 답변을 한국어로만 작성하세요.

답변:"""
        
        return self._generate_with_llm(prompt, "subjective", 5)
    
    def regenerate_korean_focused(self, question: str, domain: str) -> str:
        """한국어 중심 답변 재생성"""
        
        prompt = f"""다음 {domain} 분야 질문에 대해 순수 한국어로만 답변하세요.

{question}

한국어 전문 용어만 사용하여 명확하게 답변하세요.

답변:"""
        
        return self._generate_with_llm(prompt, "subjective", 5)
    
    def generate_simple_answer(self, prompt_text: str) -> str:
        """간단한 답변 생성"""
        
        prompt = f"다음 주제에 대해 한국어로 간단히 설명하세요: {prompt_text}"
        
        return self._generate_with_llm(prompt, "subjective", 5)
    
    def _generate_with_llm(self, prompt: str, question_type: str, max_choice: int = 5) -> str:
        """LLM 답변 생성 - 핵심 메서드"""
        
        for attempt in range(2):  # 최대 2회 시도
            try:
                # 토크나이징
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=900,
                    add_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                if self.device == "cuda":
                    inputs = inputs.to(self.model.device)
                
                # 생성 설정
                gen_config = self._get_generation_config(question_type)
                
                # 모델 실행
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config
                    )
                
                # 디코딩
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                
                # 기본 안전성 검증
                if response and len(response) > 0:
                    # 학습 데이터에 추가
                    self._add_successful_generation(prompt, response, question_type)
                    return response
                    
            except Exception as e:
                if self.verbose:
                    print(f"LLM 생성 시도 {attempt + 1} 오류: {e}")
                continue
        
        # 모든 시도 실패 시
        self._add_failed_generation(prompt, question_type)
        
        if question_type == "multiple_choice":
            import random
            return str(random.randint(1, max_choice))
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _generate_with_llm_conservative(self, prompt: str, max_choice: int = 5) -> str:
        """보수적 설정으로 LLM 생성"""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=600,
                add_special_tokens=True
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            # 매우 보수적인 설정
            gen_config = GenerationConfig(
                max_new_tokens=3,
                temperature=0.01,
                top_p=0.3,
                top_k=5,
                do_sample=True,
                repetition_penalty=1.01,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            return response
            
        except Exception:
            import random
            return str(random.randint(1, max_choice))
    
    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정 반환"""
        if question_type == "multiple_choice":
            config_dict = {
                'max_new_tokens': 3,
                'temperature': 0.1,
                'top_p': 0.7,
                'top_k': 30,
                'do_sample': True,
                'repetition_penalty': 1.02,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'early_stopping': True
            }
        else:
            config_dict = {
                'max_new_tokens': 120,
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 40,
                'do_sample': True,
                'repetition_penalty': 1.05,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'early_stopping': True,
                'length_penalty': 1.0
            }
        
        return GenerationConfig(**config_dict)
    
    def _add_successful_generation(self, prompt: str, response: str, question_type: str):
        """성공한 생성 기록"""
        success_record = {
            "prompt": prompt[:200],  # 처음 200자만 저장
            "response": response,
            "question_type": question_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.learning_data["successful_answers"].append(success_record)
        
        # 메모리 관리
        max_records = MEMORY_CONFIG['max_learning_records']['successful_answers']
        if len(self.learning_data["successful_answers"]) > max_records:
            self.learning_data["successful_answers"] = self.learning_data["successful_answers"][-max_records:]
    
    def _add_failed_generation(self, prompt: str, question_type: str):
        """실패한 생성 기록"""
        fail_record = {
            "prompt": prompt[:200],
            "question_type": question_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.learning_data["failed_answers"].append(fail_record)
        
        # 메모리 관리
        max_records = MEMORY_CONFIG['max_learning_records']['failed_answers']
        if len(self.learning_data["failed_answers"]) > max_records:
            self.learning_data["failed_answers"] = self.learning_data["failed_answers"][-max_records:]
    
    def _extract_choice_count(self, question: str) -> int:
        """질문에서 선택지 개수 추출"""
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            # 선택지 패턴: 숫자 + 공백 + 내용
            match = re.match(r'^(\d+)\s+(.+)', line.strip())
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_numbers.append(num)
        
        if choice_numbers:
            choice_numbers.sort()
            return max(choice_numbers)
        
        return 5  # 기본값
    
    def _analyze_mc_context(self, question: str, domain: str = "일반") -> Dict:
        """객관식 질문 컨텍스트 분석"""
        context = {
            "is_negative": False,
            "is_positive": False,
            "domain_hints": [],
            "key_terms": [],
            "choice_count": self._extract_choice_count(question),
            "domain": domain,
            "likely_answers": [],
            "confidence_score": 0.0
        }
        
        question_lower = question.lower()
        
        # 부정형/긍정형 판단
        for pattern in self.mc_context_patterns["negative_keywords"]:
            if re.search(pattern, question_lower):
                context["is_negative"] = True
                break
        
        for pattern in self.mc_context_patterns["positive_keywords"]:
            if re.search(pattern, question_lower):
                context["is_positive"] = True
                break
        
        return context
    
    def _process_mc_answer(self, response: str, question: str, max_choice: int, domain: str = "일반") -> str:
        """객관식 답변 처리"""
        if max_choice <= 0:
            max_choice = 5
        
        # 텍스트 정리
        response = self._clean_text_safe(response)
        
        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r'[1-9]', response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] = self.answer_distributions[max_choice].get(num, 0) + 1
                    self.mc_answer_counts[max_choice] = self.mc_answer_counts.get(max_choice, 0) + 1
                return num
        
        # 최종 폴백
        import random
        return str(random.randint(1, max_choice))
    
    def _clean_text_safe(self, text: str) -> str:
        """안전한 텍스트 정리"""
        if not text:
            return ""
        
        # 안전성 검증
        if not check_text_safety(text):
            return ""
        
        # 최소한의 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 제어 문자 제거
        text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
        
        return text
    
    # 기존 호환성 유지를 위한 메서드들
    def generate_answer(self, question: str, question_type: str, max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """답변 생성 (기존 호환성 유지)"""
        
        # 도메인 감지
        domain = self._detect_domain(question)
        
        if question_type == "multiple_choice":
            context_hint = self._analyze_mc_context(question, domain)
            pattern_hint = {"guidance": f"{domain} 분야 문제입니다."}
            return self.generate_mc_answer_with_hints(question, max_choice, domain, context_hint, pattern_hint)
        else:
            knowledge_hints = {
                "domain": domain,
                "structure_guidance": "관련 법령과 기준에 따른 관리 방안을 설명하세요.",
                "key_concepts": [domain],
                "content_direction": "전문적이고 체계적인 답변을 작성하세요."
            }
            return self.generate_subj_answer_with_knowledge(question, domain, intent_analysis or {}, knowledge_hints, None)
    
    def generate_enhanced_mc_answer(self, question: str, max_choice: int, domain: str, 
                                   pattern_hint: Dict = None, context_hint: Dict = None) -> str:
        """향상된 객관식 답변 생성 (호환성)"""
        return self.generate_mc_answer_with_hints(question, max_choice, domain, context_hint, pattern_hint)
    
    def generate_enhanced_subj_answer(self, question: str, domain: str, intent_analysis: Dict = None, template_hint: str = None) -> str:
        """향상된 주관식 답변 생성 (호환성)"""
        knowledge_hints = {
            "domain": domain,
            "structure_guidance": template_hint or "관련 법령과 기준에 따른 관리 방안을 설명하세요.",
            "key_concepts": [domain],
            "content_direction": "전문적이고 체계적인 답변을 작성하세요."
        }
        return self.generate_subj_answer_with_knowledge(question, domain, intent_analysis or {}, knowledge_hints, None)
    
    def generate_institution_answer(self, question: str, institution_hint: Dict = None, intent_analysis: Dict = None) -> str:
        """기관 답변 생성 (호환성)"""
        
        # 기관 정보 힌트 생성
        knowledge_hints = {
            "structure_guidance": "구체적인 기관명과 그 역할을 명시하세요.",
            "content_direction": "정확한 기관명과 법적 근거를 포함하세요.",
            "key_concepts": ["기관", "역할", "업무"]
        }
        
        return self.generate_subj_answer_with_knowledge(question, "일반", intent_analysis or {}, knowledge_hints, institution_hint)
    
    def _detect_domain(self, question: str) -> str:
        """도메인 감지"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["개인정보", "정보주체", "만 14세", "법정대리인"]):
            return "개인정보보호"
        elif any(word in question_lower for word in ["트로이", "악성코드", "rat", "원격제어", "딥페이크", "sbom"]):
            return "사이버보안"
        elif any(word in question_lower for word in ["전자금융", "전자적", "분쟁조정", "금융감독원"]):
            return "전자금융"
        elif any(word in question_lower for word in ["정보보안", "isms", "관리체계", "정책 수립"]):
            return "정보보안"
        elif any(word in question_lower for word in ["위험관리", "위험 관리", "재해복구", "위험수용"]):
            return "위험관리"
        elif any(word in question_lower for word in ["금융투자", "투자자문", "금융투자업"]):
            return "금융투자"
        else:
            return "일반"
    
    def _warmup(self):
        """모델 워밍업"""
        try:
            test_prompt = "테스트"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    temperature=0.01
                )
            if self.verbose:
                print("모델 워밍업 완료")
        except Exception as e:
            if self.verbose:
                print(f"워밍업 실패: {e}")
    
    def get_answer_stats(self) -> Dict:
        """답변 통계"""
        return {
            "distributions": dict(self.answer_distributions),
            "counts": dict(self.mc_answer_counts),
            "learning_data": {
                "successful_generations": len(self.learning_data["successful_answers"]),
                "failed_generations": len(self.learning_data["failed_answers"])
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 학습 데이터 저장
            self._save_learning_data()
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")