# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 학습 데이터 저장
- 질문 의도 기반 답변 생성
- 컨텍스트 활용 답변 생성
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
    MEMORY_CONFIG, get_device
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
            "choice_distribution_learning": {},
            "choice_content_analysis": {},
            "semantic_similarity_patterns": {}
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
                "choice_content_analysis": self.learning_data.get("choice_content_analysis", {}),
                "semantic_similarity_patterns": self.learning_data.get("semantic_similarity_patterns", {}),
                "last_updated": datetime.now().isoformat()
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
    def generate_mc_answer_with_context(self, question: str, max_choice: int, context_info: str) -> str:
        """컨텍스트를 활용한 객관식 답변 생성"""
        
        # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
        if max_choice <= 0:
            max_choice = 5
        
        # 컨텍스트 강화 프롬프트 생성
        prompt = self._create_context_enhanced_mc_prompt(question, max_choice, context_info)
        
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            # 생성 설정
            gen_config = self._get_generation_config("multiple_choice")
            
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
            
            # 답변 추출 및 검증
            answer = self._extract_mc_answer(response, max_choice)
            
            if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                self._record_successful_generation(question, answer, "mc_with_context", response)
                return answer
            else:
                # 재시도
                retry_answer = self._retry_mc_generation(question, max_choice, context_info)
                return retry_answer
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            return self._get_fallback_mc_answer(max_choice)
    
    def generate_subjective_answer_with_context(self, question: str, context_info: str, intent_analysis: Dict) -> str:
        """컨텍스트를 활용한 주관식 답변 생성"""
        
        # 컨텍스트 기반 프롬프트 생성
        prompt = self._create_context_enhanced_subj_prompt(question, context_info, intent_analysis)
        
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            # 생성 설정
            gen_config = self._get_generation_config("subjective")
            
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
            
            # 응답 후처리
            answer = self._process_subjective_response(response, intent_analysis)
            
            self._record_successful_generation(question, answer, "subjective_with_context", response)
            
            return answer
            
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            return self._get_fallback_subjective_answer(question, intent_analysis)
    
    def _create_context_enhanced_mc_prompt(self, question: str, max_choice: int, context_info: str) -> str:
        """컨텍스트 강화 객관식 프롬프트 생성"""
        
        # 선택지 범위 명시
        choice_range = f"1부터 {max_choice}번까지"
        
        prompts = [
            f"""다음은 금융보안 관련 객관식 문제입니다.

문제: {question}

분석 정보: {context_info}

위 정보를 참고하여 문제를 신중히 분석하고, {choice_range} 중 정답을 선택하세요.
각 선택지의 의미를 정확히 파악한 후 정답 번호만 답하세요.

정답:""",

            f"""금융보안 전문가로서 다음 문제를 해결하세요.

{question}

참고 정보: {context_info}

문제의 핵심을 파악하고 {choice_range} 중 가장 적절한 답을 선택하세요.
번호만 답하세요.

답:""",

            f"""다음 금융보안 문제를 분석하고 정답을 선택하세요.

문제: {question}

분석: {context_info}

선택지별 의미를 분석하고, 문제의 요구사항에 따라 {choice_range} 중 정답을 선택하세요.
정답 번호만 답하세요.

정답:"""
        ]
        
        return random.choice(prompts)
    
    def _create_context_enhanced_subj_prompt(self, question: str, context_info: str, intent_analysis: Dict) -> str:
        """컨텍스트 강화 주관식 프롬프트 생성"""
        
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        
        # 의도별 특화 지침
        intent_guidance = ""
        if primary_intent in self.intent_specific_prompts:
            available_prompts = self.intent_specific_prompts[primary_intent]
            intent_guidance = random.choice(available_prompts)
        else:
            intent_guidance = "다음 질문에 정확하고 상세하게 답변하세요."
        
        # 답변 유형별 추가 지침
        type_guidance = self._get_answer_type_guidance(answer_type)
        
        prompts = [
            f"""금융보안 전문가로서 다음 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}

분석 정보: {context_info}

{intent_guidance}
{type_guidance}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명

답변:""",

            f"""다음은 금융보안 분야의 전문 질문입니다. 질문의 의도를 정확히 파악하여 한국어로만 상세한 답변을 제공하세요.

질문: {question}

참고 정보: {context_info}
질문 의도: {primary_intent.replace('_', ' ')}
요구되는 답변 유형: {answer_type}

{intent_guidance}
{type_guidance}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 질문이 요구하는 정확한 내용에 집중
- 법적 근거와 실무 절차를 한국어로 설명

답변:""",

            f"""질문 분석:
- 분야 정보: {context_info}
- 의도: {primary_intent}
- 답변유형: {answer_type}

질문: {question}

위 분석을 바탕으로 다음 지침에 따라 답변하세요:

{intent_guidance}
{type_guidance}

답변 원칙:
- 한국어 전용 작성
- 의도에 정확히 부합
- 구체적이고 실무적인 내용
- 관련 법령 근거 포함

답변:"""
        ]
        
        return random.choice(prompts)
    
    def _get_answer_type_guidance(self, answer_type: str) -> str:
        """답변 유형별 지침 반환"""
        
        guidance_map = {
            "기관명": "구체적인 기관명이나 조직명을 반드시 포함하여 답변하세요. 해당 기관의 정확한 명칭과 소속을 명시하세요.",
            "특징설명": "주요 특징과 특성을 체계적으로 나열하고 설명하세요. 각 특징의 의미와 중요성을 포함하세요.",
            "지표나열": "관찰 가능한 지표와 탐지 방법을 구체적으로 제시하세요. 각 지표의 의미와 활용방법을 설명하세요.",
            "방안제시": "실무적이고 실행 가능한 대응방안을 제시하세요. 구체적인 실행 단계와 방법을 포함하세요.",
            "절차설명": "단계별 절차를 순서대로 설명하세요. 각 단계의 내용과 주의사항을 포함하세요.",
            "조치설명": "필요한 보안조치와 대응조치를 구체적으로 설명하세요.",
            "법령설명": "관련 법령과 규정을 근거로 설명하세요. 법적 근거와 요구사항을 명시하세요.",
            "정의설명": "정확한 정의와 개념을 설명하세요. 용어의 의미와 범위를 명확히 제시하세요."
        }
        
        return guidance_map.get(answer_type, "질문의 요구사항에 맞는 답변을 작성하세요.")
    
    def _extract_mc_answer(self, response: str, max_choice: int) -> str:
        """객관식 답변 추출"""
        
        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r'[1-9]', response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] += 1
                    self.mc_answer_counts[max_choice] += 1
                return num
        
        return None
    
    def _process_subjective_response(self, response: str, intent_analysis: Dict = None) -> str:
        """주관식 응답 처리"""
        
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 잘못된 인코딩으로 인한 깨진 문자 제거
        response = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 길이 제한
        if len(response) > 350:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3])
            if not response.endswith('.'):
                response += '.'
        
        # 마침표 확인
        if not response.endswith(('.', '다', '요', '함')):
            response += '.'
        
        return response
    
    def _retry_mc_generation(self, question: str, max_choice: int, context_info: str) -> str:
        """객관식 답변 재생성"""
        
        # 간소화된 프롬프트로 재시도
        simple_prompt = f"""다음 문제의 정답을 1부터 {max_choice}번 중 선택하세요.

{question}

참고: {context_info}

정답 번호만 답하세요.

답:"""
        
        try:
            inputs = self.tokenizer(simple_prompt, return_tensors="pt", truncation=True, max_length=1000)
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            answer = self._extract_mc_answer(response, max_choice)
            
            if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                return answer
            else:
                return self._get_fallback_mc_answer(max_choice)
                
        except Exception:
            return self._get_fallback_mc_answer(max_choice)
    
    def _get_fallback_mc_answer(self, max_choice: int) -> str:
        """폴백 객관식 답변"""
        
        # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
        if max_choice <= 0:
            max_choice = 5
        
        return str(random.randint(1, max_choice))
    
    def _get_fallback_subjective_answer(self, question: str, intent_analysis: Dict) -> str:
        """폴백 주관식 답변"""
        
        # 의도별 기본 답변
        primary_intent = intent_analysis.get("primary_intent", "일반") if intent_analysis else "일반"
        
        fallback_answers = {
            "기관_묻기": "관련 법령에 따라 해당 분야의 전문 기관에서 업무를 담당하고 있습니다.",
            "특징_묻기": "주요 특징은 관련 기술과 절차에 따라 체계적으로 운영되는 특성을 가집니다.",
            "지표_묻기": "주요 지표는 시스템 모니터링과 로그 분석을 통해 확인할 수 있습니다.",
            "방안_묻기": "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
            "절차_묻기": "관련 규정에 따라 단계별 절차를 수립하고 체계적으로 수행해야 합니다.",
            "조치_묻기": "적절한 보안조치를 수립하고 관련 법령에 따라 체계적으로 관리해야 합니다.",
            "일반": "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
        }
        
        return fallback_answers.get(primary_intent, fallback_answers["일반"])
    
    def _record_successful_generation(self, question: str, answer: str, generation_type: str, raw_response: str):
        """성공적인 생성 기록"""
        
        record = {
            "question": question[:200],
            "answer": answer[:300],
            "generation_type": generation_type,
            "raw_response": raw_response[:500],
            "timestamp": datetime.now().isoformat()
        }
        
        self.learning_data["successful_answers"].append(record)
        
        # 메모리 관리
        if len(self.learning_data["successful_answers"]) > MEMORY_CONFIG['max_learning_records']['successful_answers']:
            self.learning_data["successful_answers"] = self.learning_data["successful_answers"][-MEMORY_CONFIG['max_learning_records']['successful_answers']:]
    
    def generate_answer(self, question: str, question_type: str, max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """답변 생성 (레거시 호환용)"""
        
        # 도메인 감지
        domain = self._detect_domain(question)
        
        if question_type == "multiple_choice":
            # 기본 컨텍스트 구성
            context_info = f"문제 분야: {domain}"
            return self.generate_mc_answer_with_context(question, max_choice, context_info)
        else:
            # 기본 컨텍스트 구성
            context_info = f"문제 분야: {domain}"
            if intent_analysis and intent_analysis.get("primary_intent"):
                context_info += f", 질문 의도: {intent_analysis['primary_intent']}"
            
            return self.generate_subjective_answer_with_context(question, context_info, intent_analysis or {"primary_intent": "일반"})
    
    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정"""
        config_dict = GENERATION_CONFIG[question_type].copy()
        config_dict['pad_token_id'] = self.tokenizer.pad_token_id
        config_dict['eos_token_id'] = self.tokenizer.eos_token_id
        
        return GenerationConfig(**config_dict)
    
    def _detect_domain(self, question: str) -> str:
        """도메인 감지"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["개인정보", "정보주체", "만 14세", "법정대리인"]):
            return "개인정보보호"
        elif any(word in question_lower for word in ["트로이", "악성코드", "RAT", "원격제어", "딥페이크", "SBOM"]):
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
    
    def _get_domain_keywords(self, question: str) -> List[str]:
        """도메인별 키워드 반환"""
        question_lower = question.lower()
        
        if "개인정보" in question_lower:
            return ["개인정보보호법", "정보주체", "처리", "보호조치", "동의"]
        elif "전자금융" in question_lower:
            return ["전자금융거래법", "접근매체", "인증", "보안", "분쟁조정"]
        elif "보안" in question_lower or "악성코드" in question_lower:
            return ["보안정책", "탐지", "대응", "모니터링", "방어"]
        elif "금융투자" in question_lower:
            return ["자본시장법", "투자자보호", "적합성원칙", "내부통제"]
        elif "위험관리" in question_lower:
            return ["위험식별", "위험평가", "위험대응", "내부통제"]
        else:
            return ["법령", "규정", "관리", "조치", "절차"]
    
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
            "mc_accuracy_by_domain": dict(self.learning_data["mc_accuracy_by_domain"]),
            "choice_content_analysis": dict(self.learning_data.get("choice_content_analysis", {})),
            "semantic_similarity_patterns": dict(self.learning_data.get("semantic_similarity_patterns", {}))
        }
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        return {
            "successful_count": len(self.learning_data["successful_answers"]),
            "failed_count": len(self.learning_data["failed_answers"]),
            "choice_range_errors": len(self.learning_data["choice_range_errors"]),
            "question_patterns": dict(self.learning_data["question_patterns"]),
            "intent_based_answers_count": {k: len(v) for k, v in self.learning_data["intent_based_answers"].items()},
            "high_quality_templates_count": {k: len(v) for k, v in self.learning_data["high_quality_templates"].items()},
            "mc_accuracy_by_domain": dict(self.learning_data["mc_accuracy_by_domain"]),
            "avg_quality": sum(self.learning_data["answer_quality_scores"]) / len(self.learning_data["answer_quality_scores"]) if self.learning_data["answer_quality_scores"] else 0,
            "choice_content_analysis_count": len(self.learning_data.get("choice_content_analysis", {})),
            "semantic_analysis_patterns": len(self.learning_data.get("semantic_similarity_patterns", {}))
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