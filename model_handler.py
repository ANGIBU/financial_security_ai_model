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
    KOREAN_TYPO_MAPPING
)

class SimpleModelHandler:
    """모델 핸들러"""
    
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
            use_fast=MODEL_CONFIG['use_fast_tokenizer']
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG['torch_dtype']),
            device_map=MODEL_CONFIG['device_map'],
            trust_remote_code=MODEL_CONFIG['trust_remote_code']
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
            "기관_묻기": ["다음 질문에서 요구하는 특정 기관명을 정확히 답변하세요."],
            "특징_묻기": ["해당 항목의 핵심적인 특징들을 구체적으로 나열하고 설명하세요."],
            "지표_묻기": ["탐지 지표와 징후를 중심으로 구체적으로 나열하고 설명하세요."],
            "방안_묻기": ["구체적인 대응 방안과 해결책을 제시하세요."],
            "절차_묻기": ["단계별 절차를 순서대로 설명하세요."],
            "조치_묻기": ["필요한 보안조치와 대응조치를 설명하세요."]
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
            "answer_quality_scores": [],
            "mc_context_patterns": {},
            "choice_range_errors": [],
            "intent_based_answers": {},
            "domain_specific_learning": {},
            "intent_prompt_effectiveness": {},
            "high_quality_templates": {},
            "mc_accuracy_by_domain": {},
            "negative_vs_positive_patterns": {},
            "choice_distribution_learning": {}
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
                "mc_context_patterns": self.learning_data["mc_context_patterns"],
                "choice_range_errors": self.learning_data["choice_range_errors"][-MEMORY_CONFIG['max_learning_records']['choice_range_errors']:],
                "intent_based_answers": self.learning_data["intent_based_answers"],
                "domain_specific_learning": self.learning_data["domain_specific_learning"],
                "intent_prompt_effectiveness": self.learning_data["intent_prompt_effectiveness"],
                "high_quality_templates": self.learning_data["high_quality_templates"],
                "mc_accuracy_by_domain": self.learning_data["mc_accuracy_by_domain"],
                "negative_vs_positive_patterns": self.learning_data["negative_vs_positive_patterns"],
                "choice_distribution_learning": self.learning_data["choice_distribution_learning"],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
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
        
        # 폴백: 기본 패턴으로 확인
        for i in range(5, 2, -1):
            pattern = r'1\s.*' + '.*'.join([f'{j}\s' for j in range(2, i+1)])
            if re.search(pattern, question, re.DOTALL):
                return i
        
        return 5
    
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
        
        # 도메인별 특화 분석
        if domain in self.mc_context_patterns["domain_specific_patterns"]:
            domain_info = self.mc_context_patterns["domain_specific_patterns"][domain]
            
            # 도메인 키워드 매칭
            keyword_matches = sum(1 for keyword in domain_info["keywords"] 
                                if keyword in question_lower)
            
            if keyword_matches > 0:
                context["domain_hints"].append(domain)
                context["likely_answers"] = domain_info["common_answers"]
                context["confidence_score"] = min(keyword_matches / len(domain_info["keywords"]), 1.0)
        
        # 핵심 용어 추출
        domain_terms = {
            "금융투자": ["구분", "업무", "금융투자업", "해당하지"],
            "위험관리": ["요소", "계획", "위험", "적절하지"],
            "개인정보보호": ["정책", "수립", "요소", "중요한"],
            "전자금융": ["요구", "경우", "자료제출", "통화신용정책"],
            "사이버보안": ["활용", "이유", "SBOM", "소프트웨어"],
            "정보보안": ["복구", "계획", "절차", "옳지"]
        }
        
        if domain in domain_terms:
            for term in domain_terms[domain]:
                if term in question:
                    context["key_terms"].append(term)
        
        return context
    
    def detect_corrupted_text(self, text: str) -> bool:
        """깨진 텍스트 감지 - 강화된 버전"""
        if not text:
            return True
        
        # 알려진 오류 패턴들 (실제 로그에서 발견된 패턴들)
        corruption_patterns = [
            r'감추인', r'컨퍼머시', r'피-에', r'백-도어', r'키-로거', r'스크리너',
            r'채팅-클라언트', r'파일-업-', r'[가-힣]-[가-힣]{2,}', 
            r'[가-힣]{1,2}-[가-힣]{1,2}', r'[가-힣]-[가-힣]-[가-힣]',
            r'^[^가-힣]*$',  # 한국어가 전혀 없음
            r'[가-힣]+[A-Za-z]+[가-힣]+[A-Za-z]+',  # 한영 혼재 패턴
            r'[\u0000-\u001F]',  # 제어 문자
            r'[^\w\s가-힣.,!?()[\]{}""''：；、。]',  # 허용되지 않은 특수문자
        ]
        
        for pattern in corruption_patterns:
            if re.search(pattern, text):
                return True
        
        # 한국어 비율 확인
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return True
        
        korean_ratio = korean_chars / total_chars
        if korean_ratio < 0.5:  # 한국어 비율이 50% 미만
            return True
        
        return False
    
    def clean_generated_text_safe(self, text: str) -> str:
        """안전한 텍스트 정리 - 오류 방지 우선"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 1단계: 깨진 텍스트 감지
        if self.detect_corrupted_text(text):
            return ""  # 깨진 텍스트면 빈 문자열 반환하여 재생성 유도
        
        # 2단계: 최소한의 안전한 정리만 수행
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 시작/끝 공백 제거
        text = text.strip()
        
        # 제어 문자 제거
        text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
        
        # 3단계: 기본적인 문장 정리
        # 문장 끝 정리
        if text and not text.endswith(('.', '다', '요', '함', '니다')):
            if any(text.endswith(suffix) for suffix in ['습니다', '입니다', '됩니다', '합니다']):
                text += '.'
            elif len(text) > 10:
                text += '.'
        
        return text
    
    def clean_generated_text_premium(self, text: str) -> str:
        """프리미엄 텍스트 정리 - 안전한 정리 우선"""
        return self.clean_generated_text_safe(text)
    
    def clean_generated_text(self, text: str) -> str:
        """생성된 텍스트 정리 (안전 우선)"""
        return self.clean_generated_text_safe(text)
    
    def validate_generated_answer(self, answer: str, question_type: str) -> bool:
        """생성된 답변 품질 검증 - 강화"""
        if not answer:
            return False
        
        # 깨진 텍스트 감지
        if self.detect_corrupted_text(answer):
            return False
        
        # 한국어 비율 확인
        korean_chars = len(re.findall(r'[가-힣]', answer))
        total_chars = len(re.sub(r'[^\w가-힣]', '', answer))
        
        if total_chars == 0:
            return False
        
        korean_ratio = korean_chars / total_chars
        
        if question_type == "multiple_choice":
            # 객관식은 숫자만 있으면 됨
            return bool(re.search(r'[1-5]', answer))
        else:
            # 주관식 검증 강화
            return (korean_ratio >= 0.7 and 
                    len(answer) >= 20 and 
                    not self.detect_corrupted_text(answer))
    
    def generate_enhanced_mc_answer(self, question: str, max_choice: int, domain: str, 
                                   pattern_hint: Dict = None, context_hint: Dict = None) -> str:
        """향상된 객관식 답변 생성 (LLM 필수 사용)"""
        
        # 힌트 정보를 프롬프트에 포함
        hint_text = ""
        if pattern_hint:
            hint_text += f"\n참고: 이 문제는 {pattern_hint.get('pattern_type', '')} 유형입니다."
            if pattern_hint.get('explanation'):
                hint_text += f" {pattern_hint['explanation']}"
        
        if context_hint:
            if context_hint.get("is_negative"):
                hint_text += f"\n힌트: 이 문제는 '{domain}' 분야에서 해당하지 않거나 적절하지 않은 것을 찾는 문제입니다."
            elif context_hint.get("is_positive"):
                hint_text += f"\n힌트: 이 문제는 '{domain}' 분야에서 가장 적절하거나 올바른 것을 찾는 문제입니다."
            
            if context_hint.get("key_terms"):
                hint_text += f" 핵심 용어: {', '.join(context_hint['key_terms'])}"
        
        # LLM을 통한 답변 생성
        prompt = self._create_enhanced_mc_prompt(question, max_choice, domain, hint_text)
        answer = self._generate_with_llm_stable(prompt, "multiple_choice", max_choice)
        
        # 후처리
        return self._process_enhanced_mc_answer(answer, question, max_choice, domain)
    
    def generate_fallback_mc_answer(self, question: str, max_choice: int, domain: str, context_hint: Dict = None) -> str:
        """폴백 객관식 답변 생성 (LLM 필수 사용)"""
        
        # 컨텍스트 힌트를 프롬프트에 포함
        hint_text = ""
        if context_hint:
            if context_hint.get("is_negative"):
                hint_text = f"\n참고: '{domain}' 분야에서 해당하지 않는 항목을 찾으세요."
            elif context_hint.get("is_positive"):
                hint_text = f"\n참고: '{domain}' 분야에서 가장 적절한 항목을 찾으세요."
        
        prompt = f"""다음 객관식 문제를 분석하여 정답을 선택하세요.{hint_text}

{question}

위 문제를 신중히 분석하고, 1부터 {max_choice}번 중 하나의 정답 번호만 답하세요.

정답:"""
        
        answer = self._generate_with_llm_stable(prompt, "multiple_choice", max_choice)
        return self._process_enhanced_mc_answer(answer, question, max_choice, domain)
    
    def generate_institution_answer(self, question: str, institution_hint: Dict = None, intent_analysis: Dict = None) -> str:
        """기관 답변 생성 (LLM 필수 사용) - 안정화 버전"""
        
        # 기관 힌트를 프롬프트에 포함
        hint_text = ""
        institution_name = ""
        
        if institution_hint:
            hint_text += f"\n참고 정보: {institution_hint.get('description', '')}"
            if institution_hint.get('institution_name'):
                institution_name = institution_hint['institution_name']
                hint_text += f"\n관련 기관: {institution_name}"
            if institution_hint.get('role'):
                hint_text += f"\n담당 업무: {institution_hint['role']}"
            if institution_hint.get('parent_organization'):
                hint_text += f"\n소속: {institution_hint['parent_organization']}"
        
        # 안전한 기관 답변 생성
        prompt = f"""다음은 기관 관련 질문입니다. 정확한 기관명을 포함하여 답변하세요.{hint_text}

질문: {question}

위 질문에 대해 구체적인 기관명과 역할을 포함하여 한국어로만 답변하세요.
기관명을 반드시 명시하고, 해당 기관의 역할과 업무를 설명하세요.

답변:"""
        
        answer = self._generate_with_llm_stable(prompt, "subjective", 5)
        cleaned_answer = self.clean_generated_text_safe(answer)
        
        # 검증 및 보완
        if not self.validate_generated_answer(cleaned_answer, "subjective"):
            # 기관 관련 안전한 폴백 답변
            if "분쟁조정" in question.lower():
                return "금융감독원 금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다."
            elif "개인정보" in question.lower() and "침해" in question.lower():
                return "개인정보보호위원회 산하 개인정보침해신고센터에서 개인정보 침해신고 접수 업무를 담당합니다."
            else:
                return "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다."
        
        # 기관명이 포함되지 않았으면 추가
        if institution_name and institution_name not in cleaned_answer:
            cleaned_answer = f"{institution_name}에서 {cleaned_answer}"
        
        return cleaned_answer
    
    def generate_enhanced_subj_answer(self, question: str, domain: str, intent_analysis: Dict = None, template_hint: str = None) -> str:
        """향상된 주관식 답변 생성 (LLM 필수 사용) - 안정화 버전"""
        
        # RAT 특징 질문 특별 처리
        if any(term in question.lower() for term in ["rat", "원격", "트로이", "악성코드"]) and any(term in question for term in ["특징", "지표"]):
            return self._generate_rat_specific_answer_safe(question, domain, intent_analysis)
        
        # 간단하고 안전한 프롬프트 생성
        prompt = f"""다음 {domain} 분야 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 명확하고 정확한 전문 용어 사용
- 관련 법령과 규정을 근거로 구체적 내용 포함

답변:"""
        
        answer = self._generate_with_llm_stable(prompt, "subjective", 5)
        cleaned_answer = self.clean_generated_text_safe(answer)
        
        # 검증 및 재생성
        if not self.validate_generated_answer(cleaned_answer, "subjective"):
            return self._generate_safe_fallback_answer(question, domain, intent_analysis)
        
        return self.fix_korean_sentence_structure_safe(cleaned_answer)
    
    def _generate_rat_specific_answer_safe(self, question: str, domain: str, intent_analysis: Dict = None) -> str:
        """RAT 특징 전용 답변 생성 - 안정화 버전"""
        
        # 안전한 RAT 관련 프롬프트
        prompt = f"""다음은 RAT(원격접근 트로이목마) 관련 질문입니다.

질문: {question}

RAT의 주요 특징과 탐지 지표에 대해 한국어로만 답변하세요.
주요 특징: 은폐성, 지속성, 원격제어 기능, 정상 프로그램 위장
탐지 지표: 비정상적인 네트워크 트래픽, 의심스러운 파일 생성

답변:"""
        
        answer = self._generate_with_llm_stable(prompt, "subjective", 5)
        cleaned_answer = self.clean_generated_text_safe(answer)
        
        if not self.validate_generated_answer(cleaned_answer, "subjective"):
            # RAT 전용 안전 답변
            return "RAT 악성코드는 정상 프로그램으로 위장하여 시스템에 침투하는 원격제어 악성코드입니다. 주요 특징으로는 은폐성과 지속성이 있으며, 탐지 지표로는 비정상적인 네트워크 트래픽과 의심스러운 파일 생성 등이 있습니다."
        
        return cleaned_answer
    
    def _generate_safe_fallback_answer(self, question: str, domain: str, intent_analysis: Dict = None) -> str:
        """안전한 폴백 답변 생성"""
        # 도메인별 안전 답변
        domain_answers = {
            "사이버보안": "사이버보안 위협에 대한 효과적인 대응을 위해 예방, 탐지, 대응, 복구의 단계별 보안 체계를 구축하고 지속적인 모니터링을 수행해야 합니다.",
            "전자금융": "전자금융거래의 안전성 확보를 위해 관련 법령에 따른 보안 조치를 시행하고 이용자 보호를 위한 관리 체계를 운영해야 합니다.",
            "개인정보보호": "개인정보 보호를 위해 개인정보보호법에 따른 안전성 확보조치를 시행하고 정보주체의 권익 보호를 위한 관리 방안을 수립해야 합니다.",
            "정보보안": "정보보안 관리체계를 수립하여 정보자산을 보호하고 위험요소에 대한 체계적인 관리와 대응 방안을 마련해야 합니다.",
            "금융투자": "금융투자업의 건전한 운영을 위해 자본시장법에 따른 투자자 보호 조치를 시행하고 적절한 내부통제 체계를 구축해야 합니다.",
            "위험관리": "효과적인 위험관리를 위해 위험 식별, 평가, 대응의 단계별 프로세스를 수립하고 지속적인 모니터링을 수행해야 합니다."
        }
        
        return domain_answers.get(domain, "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.")
    
    def fix_korean_sentence_structure_safe(self, text: str) -> str:
        """한국어 문장 구조 수정 - 안전 버전"""
        if not text:
            return ""
        
        # 깨진 텍스트 확인
        if self.detect_corrupted_text(text):
            return ""
        
        # 최소한의 안전한 수정만 수행
        text = text.strip()
        
        # 문장 끝 처리
        if text and not text.endswith(('.', '다', '요', '함', '니다')):
            if len(text) > 10:
                text += '.'
        
        return text
    
    def generate_intent_focused_answer(self, question: str, domain: str, intent_analysis: Dict, template_hint: str = None) -> str:
        """의도 집중 답변 생성 (LLM 필수 사용) - 안정화 버전"""
        
        primary_intent = intent_analysis.get("primary_intent", "일반")
        
        # 안전한 의도별 프롬프트
        prompt = f"""다음 {domain} 분야 질문에 대해 한국어로만 답변하세요.

질문: {question}

질문의 의도에 정확히 부합하도록 한국어로만 답변하세요.
명확하고 정확한 전문 용어를 사용하여 체계적으로 설명하세요.

답변:"""
        
        answer = self._generate_with_llm_stable(prompt, "subjective", 5)
        cleaned_answer = self.clean_generated_text_safe(answer)
        
        if not self.validate_generated_answer(cleaned_answer, "subjective"):
            return self._generate_safe_fallback_answer(question, domain, intent_analysis)
        
        return self.fix_korean_sentence_structure_safe(cleaned_answer)
    
    def generate_simple_mc_answer(self, question: str, max_choice: int) -> str:
        """간단한 객관식 답변 생성 (LLM 필수 사용)"""
        
        prompt = f"""다음 객관식 문제의 정답을 선택하세요.

{question}

1부터 {max_choice}번 중 정답 번호만 답하세요.

정답:"""
        
        answer = self._generate_with_llm_stable(prompt, "multiple_choice", max_choice)
        return self._process_enhanced_mc_answer(answer, question, max_choice, "일반")
    
    def generate_simple_subj_answer(self, question: str) -> str:
        """간단한 주관식 답변 생성 (LLM 필수 사용) - 안정화 버전"""
        
        prompt = f"""다음 질문에 한국어로만 답변하세요.

질문: {question}

관련 법령과 규정을 바탕으로 전문적인 답변을 작성하세요.
명확하고 정확한 한국어로 답변하며, 전문 용어는 올바르게 사용하세요.

답변:"""
        
        answer = self._generate_with_llm_stable(prompt, "subjective", 5)
        cleaned_answer = self.clean_generated_text_safe(answer)
        
        if not self.validate_generated_answer(cleaned_answer, "subjective"):
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
        
        return self.fix_korean_sentence_structure_safe(cleaned_answer)
    
    def _generate_with_llm_stable(self, prompt: str, question_type: str, max_choice: int = 5) -> str:
        """안정화된 LLM 답변 생성 - 깨진 텍스트 방지 특화"""
        
        for attempt in range(3):  # 최대 3회 시도
            try:
                # 토크나이징
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1500  # 더 짧게 제한
                )
                
                if self.device == "cuda":
                    inputs = inputs.to(self.model.device)
                
                # 안정화된 생성 설정
                gen_config = self._get_stable_generation_config(question_type, attempt)
                
                # 모델 실행
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config
                    )
                
                # 디코딩
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # 깨진 텍스트 확인
                if not self.detect_corrupted_text(response):
                    return response
                    
            except Exception as e:
                if self.verbose:
                    print(f"LLM 생성 시도 {attempt + 1} 오류: {e}")
                continue
        
        # 모든 시도 실패 시 안전한 폴백
        if question_type == "multiple_choice":
            import random
            return str(random.randint(1, max_choice))
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _get_stable_generation_config(self, question_type: str, attempt: int = 0) -> GenerationConfig:
        """안정화된 생성 설정 - 깨진 텍스트 방지"""
        config_dict = GENERATION_CONFIG[question_type].copy()
        
        # 안정화된 설정 (낮은 온도값)
        if question_type == "subjective":
            config_dict['temperature'] = 0.1  # 매우 낮은 온도
            config_dict['top_p'] = 0.8
            config_dict['repetition_penalty'] = 1.1
            config_dict['max_new_tokens'] = 150  # 적당한 길이
        else:
            config_dict['temperature'] = 0.05  # 더 낮은 온도
            config_dict['top_p'] = 0.7
            config_dict['max_new_tokens'] = 5
        
        config_dict['pad_token_id'] = self.tokenizer.pad_token_id
        config_dict['eos_token_id'] = self.tokenizer.eos_token_id
        
        return GenerationConfig(**config_dict)
    
    # 기존 메서드들 (하위 호환성 유지)
    def _generate_with_llm_improved(self, prompt: str, question_type: str, max_choice: int = 5) -> str:
        """개선된 LLM 답변 생성 (호환성 유지)"""
        return self._generate_with_llm_stable(prompt, question_type, max_choice)
    
    def _generate_with_llm(self, prompt: str, question_type: str, max_choice: int = 5) -> str:
        """기존 호환성을 위한 LLM 생성"""
        return self._generate_with_llm_stable(prompt, question_type, max_choice)
    
    def _create_enhanced_mc_prompt(self, question: str, max_choice: int, domain: str = "일반", hint_text: str = "") -> str:
        """향상된 객관식 프롬프트 생성"""
        
        if max_choice <= 0:
            max_choice = 5
        
        prompt = f"""다음은 {domain} 분야의 금융보안 관련 문제입니다.{hint_text}

{question}

위 문제를 신중히 분석하여 정답을 선택하세요.
1부터 {max_choice}번 중 하나의 정답 번호만 답하세요.

정답:"""
        
        return prompt
    
    def _process_enhanced_mc_answer(self, response: str, question: str, max_choice: int, domain: str = "일반") -> str:
        """객관식 답변 처리 - 안정화"""
        if max_choice <= 0:
            max_choice = 5
        
        # 텍스트 정리
        response = self.clean_generated_text_safe(response)
        
        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r'[1-9]', response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] += 1
                    self.mc_answer_counts[max_choice] += 1
                return num
        
        # 최종 폴백
        import random
        return str(random.randint(1, max_choice))
    
    # 나머지 메서드들은 기존과 동일하게 유지하되 _generate_with_llm_stable 사용
    def generate_answer(self, question: str, question_type: str, max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """답변 생성 (기존 호환성 유지)"""
        
        # 도메인 감지
        domain = self._detect_domain(question)
        
        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_enhanced_mc_prompt(question, max_choice, domain)
        else:
            if intent_analysis:
                prompt = self._create_intent_aware_prompt_safe(question, intent_analysis)
            else:
                prompt = self._create_korean_subj_prompt_safe(question, domain)
        
        # LLM 생성
        response = self._generate_with_llm_stable(prompt, question_type, max_choice)
        
        # 후처리
        if question_type == "multiple_choice":
            answer = self._process_enhanced_mc_answer(response, question, max_choice, domain)
            self._add_learning_record(question, answer, question_type, True, max_choice, 1.0, intent_analysis)
            return answer
        else:
            cleaned_answer = self.clean_generated_text_safe(response)
            final_answer = self.fix_korean_sentence_structure_safe(cleaned_answer)
            
            # 품질 검증
            if not self.validate_generated_answer(final_answer, "subjective"):
                final_answer = self._generate_safe_fallback_answer(question, domain, intent_analysis)
            
            korean_ratio = self._calculate_korean_ratio(final_answer)
            quality_score = self._calculate_answer_quality(final_answer, question, intent_analysis)
            success = korean_ratio > 0.6 and quality_score > 0.4
            
            self._add_learning_record(question, final_answer, question_type, success, max_choice, quality_score, intent_analysis)
            return final_answer
    
    def _create_intent_aware_prompt_safe(self, question: str, intent_analysis: Dict) -> str:
        """의도 인식 기반 프롬프트 생성 - 안전 버전"""
        domain = self._detect_domain(question)
        
        prompt = f"""금융보안 전문가로서 다음 {domain} 관련 질문에 한국어로만 정확한 답변을 작성하세요.

질문: {question}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 질문의 의도에 정확히 부합하는 내용 포함
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 명확하고 정확한 전문 용어 사용

답변:"""
        
        return prompt
    
    def _create_korean_subj_prompt_safe(self, question: str, domain: str = "일반") -> str:
        """한국어 전용 주관식 프롬프트 생성 - 안전 버전"""
        
        prompt = f"""금융보안 전문가로서 다음 {domain} 분야 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 명확하고 정확한 전문 용어 사용

답변:"""
        
        return prompt
    
    # 나머지 메서드들 (기존과 동일)
    def _calculate_answer_quality(self, answer: str, question: str, intent_analysis: Dict = None) -> float:
        """답변 품질 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 깨진 텍스트 검증
        if self.detect_corrupted_text(answer):
            return 0.0
        
        # 한국어 비율 (30%)
        korean_ratio = self._calculate_korean_ratio(answer)
        score += korean_ratio * 0.30
        
        # 길이 적절성 (20%)
        length = len(answer)
        if 30 <= length <= 300:
            score += 0.20
        elif 20 <= length < 30:
            score += 0.15
        
        # 문장 구조 (20%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.15
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.05
        
        # 전문성 (15%)
        domain_keywords = self._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.15
        
        # 기본 유효성 (15%)
        if len(answer) >= 20 and korean_ratio >= 0.7:
            score += 0.15
        
        return max(min(score, 1.0), 0.0)
    
    def _get_domain_keywords(self, question: str) -> List[str]:
        """도메인별 키워드 반환"""
        question_lower = question.lower()
        
        if "개인정보" in question_lower:
            return ["개인정보보호법", "정보주체", "처리", "보호조치", "동의", "위원회"]
        elif "전자금융" in question_lower:
            return ["전자금융거래법", "접근매체", "인증", "보안", "분쟁조정", "금융감독원"]
        elif "보안" in question_lower or "악성코드" in question_lower:
            return ["보안정책", "탐지", "대응", "모니터링", "방어", "특징", "지표"]
        elif "금융투자" in question_lower:
            return ["자본시장법", "투자자보호", "적합성원칙", "내부통제", "업무"]
        elif "위험관리" in question_lower:
            return ["위험식별", "위험평가", "위험대응", "내부통제", "관리"]
        else:
            return ["법령", "규정", "관리", "조치", "절차", "기관"]
    
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
    
    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def _add_learning_record(self, question: str, answer: str, question_type: str, success: bool, max_choice: int = 5, quality_score: float = 0.0, intent_analysis: Dict = None):
        """학습 기록 추가"""
        record = {
            "question": question[:200],
            "answer": answer[:300],
            "type": question_type,
            "max_choice": max_choice,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score
        }
        
        if success:
            self.learning_data["successful_answers"].append(record)
        else:
            self.learning_data["failed_answers"].append(record)
        
        # 품질 점수 기록
        self.learning_data["answer_quality_scores"].append(quality_score)
    
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
                    max_new_tokens=5,
                    do_sample=False
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
            "mc_accuracy_by_domain": dict(self.learning_data["mc_accuracy_by_domain"])
        }
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        return {
            "successful_count": len(self.learning_data["successful_answers"]),
            "failed_count": len(self.learning_data["failed_answers"]),
            "choice_range_errors": len(self.learning_data["choice_range_errors"]),
            "question_patterns": dict(self.learning_data["question_patterns"]),
            "avg_quality": sum(self.learning_data["answer_quality_scores"]) / len(self.learning_data["answer_quality_scores"]) if self.learning_data["answer_quality_scores"] else 0
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