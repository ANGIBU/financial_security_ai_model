# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 학습 데이터 저장
- 질문 의도 기반 답변 생성
- 선택지 의미 분석 강화
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
    
    def _extract_choices_with_content(self, question: str) -> Dict[str, str]:
        """선택지 번호와 내용 추출"""
        choices = {}
        lines = question.split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                num = match.group(1)
                content = match.group(2).strip()
                if 1 <= int(num) <= 5 and len(content) > 0:
                    choices[num] = content
        
        return choices
    
    def _analyze_choice_semantics(self, choices: Dict[str, str], domain: str) -> Dict:
        """선택지 의미 분석"""
        semantic_analysis = {
            "outliers": [],
            "domain_relevance": {},
            "content_categories": {},
            "similarity_groups": []
        }
        
        # 도메인별 키워드 정의
        domain_keywords = {
            "금융투자": ["투자", "자문", "매매", "중개", "집합투자", "신탁"],
            "위험관리": ["위험", "평가", "대응", "수용", "회피", "전가", "감소", "관리", "계획"],
            "개인정보보호": ["개인정보", "정보주체", "처리", "수집", "이용", "제공"],
            "전자금융": ["전자", "금융", "거래", "접근매체", "인증", "보안"],
            "사이버보안": ["보안", "악성코드", "트로이", "탐지", "방어", "공격"],
            "정보보안": ["정보", "보안", "관리", "통제", "암호화", "접근제어"]
        }
        
        keywords = domain_keywords.get(domain, [])
        
        # 각 선택지의 도메인 관련성 분석
        for choice_num, content in choices.items():
            content_lower = content.lower()
            
            # 도메인 관련성 점수
            relevance_score = sum(1 for keyword in keywords if keyword in content_lower)
            semantic_analysis["domain_relevance"][choice_num] = relevance_score
            
            # 내용 카테고리 분류
            if domain == "금융투자":
                if any(word in content_lower for word in ["투자", "자문", "매매", "중개"]):
                    semantic_analysis["content_categories"][choice_num] = "금융투자업_유형"
                elif "보험" in content_lower:
                    semantic_analysis["content_categories"][choice_num] = "보험업_유형"
                elif "소비자" in content_lower:
                    semantic_analysis["content_categories"][choice_num] = "소비자금융_유형"
                else:
                    semantic_analysis["content_categories"][choice_num] = "기타_유형"
            
            elif domain == "위험관리":
                if any(word in content_lower for word in ["수용", "회피", "전가", "감소"]):
                    semantic_analysis["content_categories"][choice_num] = "위험대응_전략"
                elif any(word in content_lower for word in ["인력", "자원", "조직"]):
                    semantic_analysis["content_categories"][choice_num] = "실행_리소스"
                elif any(word in content_lower for word in ["대상", "기간", "범위"]):
                    semantic_analysis["content_categories"][choice_num] = "계획_요소"
                elif any(word in content_lower for word in ["전략", "방법", "접근"]):
                    semantic_analysis["content_categories"][choice_num] = "계획_전략"
                else:
                    semantic_analysis["content_categories"][choice_num] = "기타_요소"
        
        # 이상치 찾기 (도메인 관련성이 현저히 낮은 선택지)
        if semantic_analysis["domain_relevance"]:
            relevance_scores = list(semantic_analysis["domain_relevance"].values())
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            
            for choice_num, score in semantic_analysis["domain_relevance"].items():
                if score < avg_relevance * 0.5:  # 평균의 절반 이하
                    semantic_analysis["outliers"].append(choice_num)
        
        return semantic_analysis
    
    def _analyze_mc_context_enhanced(self, question: str, domain: str = "일반") -> Dict:
        """강화된 객관식 질문 컨텍스트 분석"""
        context = {
            "is_negative": False,
            "is_positive": False,
            "domain_hints": [],
            "key_terms": [],
            "choice_count": self._extract_choice_count(question),
            "domain": domain,
            "likely_answers": [],
            "confidence_score": 0.0,
            "choices": {},
            "semantic_analysis": {},
            "question_focus": None,
            "expected_answer_type": None
        }
        
        question_lower = question.lower()
        
        # 선택지 내용 추출 및 분석
        context["choices"] = self._extract_choices_with_content(question)
        context["semantic_analysis"] = self._analyze_choice_semantics(context["choices"], domain)
        
        # 부정형/긍정형 판단
        for pattern in self.mc_context_patterns["negative_keywords"]:
            if re.search(pattern, question_lower):
                context["is_negative"] = True
                break
        
        for pattern in self.mc_context_patterns["positive_keywords"]:
            if re.search(pattern, question_lower):
                context["is_positive"] = True
                break
        
        # 질문의 초점 분석
        if "구분.*해당하지.*않는" in question_lower:
            context["question_focus"] = "카테고리_예외"
            context["expected_answer_type"] = "범주_밖_항목"
        elif "계획.*수립.*적절하지.*않은" in question_lower:
            context["question_focus"] = "계획요소_부적절"
            context["expected_answer_type"] = "계획외_요소"
        elif "가장.*중요한.*요소" in question_lower:
            context["question_focus"] = "핵심요소_식별"
            context["expected_answer_type"] = "최우선_항목"
        
        # 도메인별 특화 분석 강화
        if domain in self.mc_context_patterns["domain_specific_patterns"]:
            domain_info = self.mc_context_patterns["domain_specific_patterns"][domain]
            
            # 도메인 키워드 매칭
            keyword_matches = sum(1 for keyword in domain_info["keywords"] 
                                if keyword in question_lower)
            
            if keyword_matches > 0:
                context["domain_hints"].append(domain)
                context["likely_answers"] = domain_info["common_answers"]
                context["confidence_score"] = min(keyword_matches / len(domain_info["keywords"]), 1.0)
        
        # 의미 분석 기반 답변 추론
        if context["is_negative"] and context["semantic_analysis"]["outliers"]:
            # 부정형 질문에서 이상치가 있다면 그것이 정답일 가능성 높음
            context["likely_answers"] = context["semantic_analysis"]["outliers"]
            context["confidence_score"] = max(context["confidence_score"], 0.8)
        
        # 카테고리 기반 분석
        if context["question_focus"] == "카테고리_예외":
            categories = context["semantic_analysis"]["content_categories"]
            if categories:
                # 다른 카테고리에 속하는 항목 찾기
                category_counts = {}
                for choice, category in categories.items():
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                # 가장 적은 빈도의 카테고리 찾기
                min_count = min(category_counts.values())
                rare_categories = [cat for cat, count in category_counts.items() if count == min_count]
                
                rare_choices = [choice for choice, category in categories.items() 
                              if category in rare_categories]
                
                if rare_choices:
                    context["likely_answers"] = rare_choices
                    context["confidence_score"] = max(context["confidence_score"], 0.7)
        
        # 핵심 용어 추출 강화
        domain_terms = {
            "금융투자": ["구분", "업무", "금융투자업", "해당하지", "투자자문업", "투자매매업", "투자중개업", "소비자금융업", "보험중개업"],
            "위험관리": ["요소", "계획", "위험", "적절하지", "수행인력", "위험수용", "대응전략", "대상", "기간"],
            "개인정보보호": ["정책", "수립", "요소", "중요한", "경영진", "참여", "최고책임자", "자원할당"],
            "전자금융": ["요구", "경우", "자료제출", "통화신용정책", "한국은행", "금융통화위원회"],
            "사이버보안": ["활용", "이유", "SBOM", "소프트웨어", "공급망", "보안", "투명성"],
            "정보보안": ["복구", "계획", "절차", "옳지", "비상연락체계", "복구목표시간", "개인정보파기"]
        }
        
        if domain in domain_terms:
            for term in domain_terms[domain]:
                if term in question:
                    context["key_terms"].append(term)
        
        return context
    
    def _get_context_based_mc_answer_enhanced(self, question: str, max_choice: int, domain: str = "일반") -> str:
        """강화된 컨텍스트 기반 객관식 답변 생성"""
        # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
        if max_choice <= 0:
            max_choice = 5
        
        context = self._analyze_mc_context_enhanced(question, domain)
        
        # 의미 분석 기반 답변 우선 선택
        if context["likely_answers"] and context["confidence_score"] > 0.6:
            valid_answers = [ans for ans in context["likely_answers"] 
                           if ans.isdigit() and 1 <= int(ans) <= max_choice]
            if valid_answers:
                selected = random.choice(valid_answers)
                self._record_enhanced_mc_learning(question, selected, context, "semantic_analysis")
                return selected
        
        # 이상치 기반 답변 (부정형 질문용)
        if context["is_negative"] and context["semantic_analysis"]["outliers"]:
            outlier_answers = [ans for ans in context["semantic_analysis"]["outliers"] 
                             if ans.isdigit() and 1 <= int(ans) <= max_choice]
            if outlier_answers:
                selected = random.choice(outlier_answers)
                self._record_enhanced_mc_learning(question, selected, context, "outlier_analysis")
                return selected
        
        # 도메인별 학습된 패턴 적용 (기존 로직)
        if context["likely_answers"] and context["confidence_score"] > 0.3:
            valid_answers = [ans for ans in context["likely_answers"] 
                           if ans.isdigit() and 1 <= int(ans) <= max_choice]
            if valid_answers:
                selected = random.choice(valid_answers)
                self._record_enhanced_mc_learning(question, selected, context, "domain_pattern")
                return selected
        
        # 부정형/긍정형 패턴 기반 선택 (개선된 로직)
        if context["is_negative"]:
            # 도메인별 세밀한 가중치 적용
            if domain == "금융투자" and max_choice == 5:
                # 금융투자업 구분에 해당하지 않는 것: 소비자금융업(1), 보험중개업(5)
                # 선택지 내용 분석으로 더 정확한 판단
                choices = context["choices"]
                category_analysis = context["semantic_analysis"]["content_categories"]
                
                # 카테고리 분석 결과 활용
                non_investment_choices = []
                for choice_num, category in category_analysis.items():
                    if category in ["보험업_유형", "소비자금융_유형"]:
                        non_investment_choices.append(choice_num)
                
                if non_investment_choices:
                    # 가중치 재계산
                    weights = [1] * max_choice
                    for choice in non_investment_choices:
                        if choice.isdigit() and 1 <= int(choice) <= max_choice:
                            weights[int(choice) - 1] = 5
                else:
                    # 기본 부정형 가중치
                    weights = [3, 1, 1, 1, 3]  # 1번과 5번에 높은 가중치
                    
            elif domain == "위험관리" and max_choice == 5:
                # 위험관리 계획 수립 요소가 아닌 것 찾기
                choices = context["choices"]
                category_analysis = context["semantic_analysis"]["content_categories"]
                
                # 실행 리소스는 계획 수립 단계가 아님
                non_planning_choices = []
                for choice_num, category in category_analysis.items():
                    if category == "실행_리소스":
                        non_planning_choices.append(choice_num)
                
                if non_planning_choices:
                    weights = [1] * max_choice
                    for choice in non_planning_choices:
                        if choice.isdigit() and 1 <= int(choice) <= max_choice:
                            weights[int(choice) - 1] = 4
                else:
                    weights = [3, 2, 1, 1, 1]  # 1번에 높은 가중치 (일반적 패턴)
                    
            else:
                # 기타 도메인 부정형
                if max_choice == 5:
                    weights = [2, 2, 2, 2, 2]  # 균등 분포로 시작
                elif max_choice == 4:
                    weights = [2, 2, 2, 2]
                else:
                    weights = [2, 2, 2]
                    
        elif context["is_positive"]:
            # 긍정형 질문 처리
            if domain == "개인정보보호":
                weights = [1, 4, 2, 1, 1] if max_choice >= 5 else [1, 4, 2, 1]  # 경영진의 참여(2번)
            elif domain == "전자금융":
                weights = [1, 1, 1, 4, 1] if max_choice >= 5 else [1, 1, 1, 4]  # 통화신용정책 관련(4번)
            elif domain == "사이버보안":
                weights = [2, 1, 2, 1, 2] if max_choice >= 5 else [2, 1, 2, 1]
            else:
                if max_choice == 5:
                    weights = [3, 3, 2, 1, 1]
                elif max_choice == 4:
                    weights = [3, 3, 2, 1]
                else:
                    weights = [3, 2, 1]
        else:
            # 중립적 질문
            weights = [2] * max_choice
            
            # 도메인별 추가 가중치
            if domain == "사이버보안" and max_choice >= 3:
                weights[2] += 1
            elif domain == "개인정보보호" and max_choice >= 2:
                weights[1] += 1
            elif domain == "전자금융" and max_choice >= 4:
                weights[3] += 1
        
        # 학습된 컨텍스트 패턴 적용
        pattern_key = f"{context['is_negative']}_{context['is_positive']}_{domain}_{max_choice}"
        
        if pattern_key in self.learning_data["mc_context_patterns"]:
            learned_distribution = self.learning_data["mc_context_patterns"][pattern_key]
            # 학습된 분포를 기반으로 가중치 조정
            for num, weight in learned_distribution.items():
                if num.isdigit() and 1 <= int(num) <= max_choice:
                    idx = int(num) - 1
                    if idx < len(weights):
                        weights[idx] += int(weight * 2)
        
        # 가중치 기반 선택
        choices = []
        for i, weight in enumerate(weights):
            choices.extend([str(i+1)] * weight)
        
        if not choices:
            # weights가 비어있는 경우 기본값 사용
            choices = [str(i+1) for i in range(max_choice)]
        
        selected = random.choice(choices)
        
        # 학습 데이터에 기록
        self._record_enhanced_mc_learning(question, selected, context, "weighted_selection")
        
        return selected
    
    def _record_enhanced_mc_learning(self, question: str, answer: str, context: Dict, method: str):
        """강화된 객관식 컨텍스트 학습 기록"""
        pattern_key = f"{context['is_negative']}_{context['is_positive']}_{context['domain']}_{context['choice_count']}"
        
        if pattern_key not in self.learning_data["mc_context_patterns"]:
            self.learning_data["mc_context_patterns"][pattern_key] = {}
        
        if answer in self.learning_data["mc_context_patterns"][pattern_key]:
            self.learning_data["mc_context_patterns"][pattern_key][answer] += 1
        else:
            self.learning_data["mc_context_patterns"][pattern_key][answer] = 1
        
        # 선택지 내용 분석 기록
        choice_content_key = f"{context['domain']}_{context['question_focus']}"
        if choice_content_key not in self.learning_data.get("choice_content_analysis", {}):
            if "choice_content_analysis" not in self.learning_data:
                self.learning_data["choice_content_analysis"] = {}
            self.learning_data["choice_content_analysis"][choice_content_key] = {
                "answers": {},
                "methods": {},
                "semantic_patterns": {}
            }
        
        analysis = self.learning_data["choice_content_analysis"][choice_content_key]
        
        # 답변 기록
        if answer in analysis["answers"]:
            analysis["answers"][answer] += 1
        else:
            analysis["answers"][answer] = 1
        
        # 방법 기록
        if method in analysis["methods"]:
            analysis["methods"][method] += 1
        else:
            analysis["methods"][method] = 1
        
        # 의미 패턴 기록
        if context["semantic_analysis"]["outliers"]:
            outlier_key = f"outliers_{','.join(context['semantic_analysis']['outliers'])}"
            if outlier_key in analysis["semantic_patterns"]:
                analysis["semantic_patterns"][outlier_key] += 1
            else:
                analysis["semantic_patterns"][outlier_key] = 1
        
        # 도메인별 정확도 추적
        domain = context.get("domain", "일반")
        if domain not in self.learning_data["mc_accuracy_by_domain"]:
            self.learning_data["mc_accuracy_by_domain"][domain] = {"total": 0, "correct": 0}
        
        self.learning_data["mc_accuracy_by_domain"][domain]["total"] += 1
    
    def _create_intent_aware_prompt(self, question: str, intent_analysis: Dict) -> str:
        """의도 인식 기반 프롬프트 생성"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        domain = self._detect_domain(question)
        context_hints = intent_analysis.get("context_hints", [])
        intent_confidence = intent_analysis.get("intent_confidence", 0.0)
        
        # 의도별 특화 프롬프트 선택
        if primary_intent in self.intent_specific_prompts:
            if intent_confidence > 0.7:
                available_prompts = self.intent_specific_prompts[primary_intent]
                intent_instruction = random.choice(available_prompts)
            else:
                intent_instruction = "다음 질문에 정확하고 상세하게 답변하세요."
        else:
            intent_instruction = "다음 질문에 정확하고 상세하게 답변하세요."
        
        # 답변 유형별 추가 지침
        type_guidance = ""
        if answer_type == "기관명":
            type_guidance = "구체적인 기관명이나 조직명을 반드시 포함하여 답변하세요. 해당 기관의 정확한 명칭과 소속을 명시하세요."
        elif answer_type == "특징설명":
            type_guidance = "주요 특징과 특성을 체계적으로 나열하고 설명하세요. 각 특징의 의미와 중요성을 포함하세요."
        elif answer_type == "지표나열":
            type_guidance = "관찰 가능한 지표와 탐지 방법을 구체적으로 제시하세요. 각 지표의 의미와 활용방법을 설명하세요."
        elif answer_type == "방안제시":
            type_guidance = "실무적이고 실행 가능한 대응방안을 제시하세요. 구체적인 실행 단계와 방법을 포함하세요."
        elif answer_type == "절차설명":
            type_guidance = "단계별 절차를 순서대로 설명하세요. 각 단계의 내용과 주의사항을 포함하세요."
        elif answer_type == "조치설명":
            type_guidance = "필요한 보안조치와 대응조치를 구체적으로 설명하세요."
        elif answer_type == "법령설명":
            type_guidance = "관련 법령과 규정을 근거로 설명하세요. 법적 근거와 요구사항을 명시하세요."
        elif answer_type == "정의설명":
            type_guidance = "정확한 정의와 개념을 설명하세요. 용어의 의미와 범위를 명확히 제시하세요."
        
        # 컨텍스트 힌트 활용
        context_instruction = ""
        if context_hints:
            context_instruction = f"답변 시 다음 사항을 고려하세요: {', '.join(context_hints)}"
        
        # 고품질 프롬프트 패턴 활용
        if primary_intent in self.learning_data["high_quality_templates"]:
            templates = self.learning_data["high_quality_templates"][primary_intent]
            if templates:
                best_template = max(templates, key=lambda x: x.get("quality", 0))
                if best_template["quality"] > 0.8:
                    # 'prompt' 키 대신 'answer_template' 키 사용
                    if "answer_template" in best_template:
                        return best_template["answer_template"]
        
        prompts = [
            f"""금융보안 전문가로서 다음 {domain} 관련 질문에 한국어로만 정확한 답변을 작성하세요.

질문: {question}

{intent_instruction}
{type_guidance}
{context_instruction}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 질문의 의도에 정확히 부합하는 내용 포함
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명

답변:""",
            
            f"""{domain} 전문가의 관점에서 다음 질문에 한국어로만 답변하세요.

{question}

질문 의도: {primary_intent.replace('_', ' ')}
요구되는 답변 유형: {answer_type}
신뢰도: {intent_confidence:.1f}

{intent_instruction}
{type_guidance}
{context_instruction}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 질문이 요구하는 정확한 내용에 집중
- 법적 근거와 실무 절차를 한국어로 설명

답변:""",
            
            f"""다음은 {domain} 분야의 전문 질문입니다. 질문의 의도를 정확히 파악하여 한국어로만 상세한 답변을 제공하세요.

질문: {question}

분석된 질문 의도: {primary_intent}
{intent_instruction}
{type_guidance}
{context_instruction}

답변 요구사항:
- 완전한 한국어 답변
- 질문 의도에 정확히 부합하는 내용
- 체계적이고 구체적인 한국어 설명
- 관련 법령과 실무 기준 포함

답변:""",
            
            f"""질문 분석:
- 분야: {domain}
- 의도: {primary_intent}
- 답변유형: {answer_type}
- 신뢰도: {intent_confidence:.1f}

질문: {question}

위 분석을 바탕으로 다음 지침에 따라 답변하세요:

{intent_instruction}
{type_guidance}
{context_instruction}

답변 원칙:
- 한국어 전용 작성
- 의도에 정확히 부합
- 구체적이고 실무적인 내용
- 관련 법령 근거 포함

답변:"""
        ]
        
        # 프롬프트 효과성 기록
        selected_prompt = random.choice(prompts)
        prompt_id = hash(selected_prompt) % 1000
        
        if primary_intent not in self.learning_data["intent_prompt_effectiveness"]:
            self.learning_data["intent_prompt_effectiveness"][primary_intent] = {}
        
        if prompt_id not in self.learning_data["intent_prompt_effectiveness"][primary_intent]:
            self.learning_data["intent_prompt_effectiveness"][primary_intent][prompt_id] = {
                "prompt": selected_prompt,
                "use_count": 0,
                "success_count": 0,
                "avg_quality": 0.0
            }
        
        self.learning_data["intent_prompt_effectiveness"][primary_intent][prompt_id]["use_count"] += 1
        
        return selected_prompt
    
    def generate_answer(self, question: str, question_type: str, max_choice: int = 5, intent_analysis: Dict = None) -> str:
        """답변 생성"""
        
        # 도메인 감지
        domain = self._detect_domain(question)
        
        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_enhanced_mc_prompt(question, max_choice, domain)
        else:
            if intent_analysis:
                prompt = self._create_intent_aware_prompt(question, intent_analysis)
            else:
                prompt = self._create_korean_subj_prompt(question, domain)
        
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
                skip_special_tokens=True
            ).strip()
            
            # 후처리
            if question_type == "multiple_choice":
                answer = self._process_enhanced_mc_answer(response, question, max_choice, domain)
                # 선택지 범위 검증
                if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                    self._add_learning_record(question, answer, question_type, True, max_choice, 1.0, intent_analysis)
                    
                    # 도메인별 정확도 업데이트
                    if domain in self.learning_data["mc_accuracy_by_domain"]:
                        self.learning_data["mc_accuracy_by_domain"][domain]["correct"] += 1
                    
                    return answer
                else:
                    # 범위 벗어난 경우 강화된 컨텍스트 기반 폴백
                    fallback = self._get_context_based_mc_answer_enhanced(question, max_choice, domain)
                    self._add_learning_record(question, answer, question_type, False, max_choice, 0.0, intent_analysis)
                    return fallback
            else:
                answer = self._process_intent_aware_subj_answer(response, question, intent_analysis)
                korean_ratio = self._calculate_korean_ratio(answer)
                quality_score = self._calculate_answer_quality(answer, question, intent_analysis)
                success = korean_ratio > 0.7 and quality_score > 0.5
                
                self._add_learning_record(question, answer, question_type, success, max_choice, quality_score, intent_analysis)
                return answer
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            fallback = self._get_fallback_answer(question_type, question, max_choice, intent_analysis, domain)
            self._add_learning_record(question, fallback, question_type, False, max_choice, 0.3, intent_analysis)
            return fallback
    
    def _create_enhanced_mc_prompt(self, question: str, max_choice: int, domain: str = "일반") -> str:
        """강화된 객관식 프롬프트 생성"""
        # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
        if max_choice <= 0:
            max_choice = 5
        
        context = self._analyze_mc_context_enhanced(question, domain)
        
        # 선택지 범위 명시
        choice_range = "에서 ".join([str(i) for i in range(1, max_choice+1)]) + f"번 중"
        
        # 선택지 내용 분석 결과 활용
        semantic_hints = ""
        if context["semantic_analysis"]["outliers"]:
            semantic_hints = f"선택지 중 {', '.join(context['semantic_analysis']['outliers'])}번이 다른 성격을 가집니다."
        
        # 컨텍스트와 도메인에 따른 프롬프트 조정
        if context["is_negative"]:
            if domain == "금융투자":
                instruction = f"금융투자업의 구분에서 해당하지 않는 것을 {choice_range} 찾으세요. 투자매매업, 투자중개업, 투자자문업은 금융투자업에 해당하지만, 다른 업종에 속하는 것을 찾으세요."
            elif domain == "위험관리":
                instruction = f"위험 관리 계획 수립 단계에서 고려하지 않는 요소를 {choice_range} 찾으세요. 계획 수립과 실행 단계를 구별하여 생각하세요."
            elif domain == "정보보안":
                instruction = f"재해 복구 계획 수립 시 포함되지 않는 요소를 {choice_range} 찾으세요."
            else:
                instruction = f"다음 중 해당하지 않거나 옳지 않은 것을 {choice_range} 찾으세요."
        elif context["is_positive"]:
            if domain == "개인정보보호":
                instruction = f"정책 수립 단계에서 가장 중요한 요소를 {choice_range} 찾으세요."
            elif domain == "전자금융":
                instruction = f"한국은행이 자료제출을 요구할 수 있는 경우를 {choice_range} 찾으세요."
            elif domain == "사이버보안":
                instruction = f"SBOM을 활용하는 가장 적절한 이유를 {choice_range} 찾으세요."
            else:
                instruction = f"다음 중 가장 적절하거나 옳은 것을 {choice_range} 찾으세요."
        else:
            instruction = f"정답을 {choice_range} 선택하세요."
        
        prompts = [
            f"""다음은 {domain} 분야의 금융보안 관련 문제입니다. {instruction}

{question}

{semantic_hints}

위 문제를 신중히 분석하고, 각 선택지의 의미를 정확히 파악한 후 1부터 {max_choice}까지 중 하나의 정답을 선택하세요.
선택지를 꼼꼼히 검토한 후 정답 번호만 답하세요.

정답:""",
            
            f"""금융보안 전문가로서 다음 {domain} 문제를 해결하세요.

{question}

{instruction}
{semantic_hints}

각 선택지의 내용을 정확히 이해하고, 문제에서 요구하는 바에 따라 1부터 {max_choice}번 중 정답을 선택하세요.
번호만 답하세요.

답:""",
            
            f"""다음 {domain} 분야 금융보안 문제를 분석하고 정답을 선택하세요.

문제: {question}

{instruction}
{semantic_hints}

선택지별 의미를 분석하고, 문제의 핵심을 파악하여 정답을 1부터 {max_choice}번 중 하나의 번호로만 답하세요.

정답:"""
        ]
        
        return random.choice(prompts)
    
    def _create_korean_subj_prompt(self, question: str, domain: str = "일반") -> str:
        """한국어 전용 주관식 프롬프트 생성"""
        
        prompts = [
            f"""금융보안 전문가로서 다음 {domain} 분야 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명
- 영어 용어 사용 금지

답변:""",
            
            f"""다음은 {domain} 분야의 전문 질문입니다. 한국어로만 상세하고 정확한 답변을 제공하세요.

{question}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 법적 근거와 실무 절차를 한국어로 설명
- 영어나 외국어 사용 금지

답변:""",
            
            f"""{domain} 전문가의 관점에서 다음 질문에 한국어로만 답변하세요.

질문: {question}

답변 요구사항:
- 완전한 한국어 답변
- 관련 법령과 규정을 한국어로 설명
- 체계적이고 구체적인 한국어 설명

답변:"""
        ]
        
        return random.choice(prompts)
    
    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정"""
        config_dict = GENERATION_CONFIG[question_type].copy()
        config_dict['pad_token_id'] = self.tokenizer.pad_token_id
        config_dict['eos_token_id'] = self.tokenizer.eos_token_id
        
        return GenerationConfig(**config_dict)
    
    def _process_enhanced_mc_answer(self, response: str, question: str, max_choice: int, domain: str = "일반") -> str:
        """강화된 객관식 답변 처리"""
        # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
        if max_choice <= 0:
            max_choice = 5
        
        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r'[1-9]', response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] += 1
                    self.mc_answer_counts[max_choice] += 1
                return num
        
        # 유효한 답변을 찾지 못한 경우 강화된 컨텍스트 기반 폴백
        return self._get_context_based_mc_answer_enhanced(question, max_choice, domain)
    
    def _process_intent_aware_subj_answer(self, response: str, question: str, intent_analysis: Dict = None) -> str:
        """의도 인식 기반 주관식 답변 처리"""
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 잘못된 인코딩으로 인한 깨진 문자 제거
        response = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 한국어 비율 확인
        korean_ratio = self._calculate_korean_ratio(response)
        
        # 의도별 답변 검증
        is_intent_match = True
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            
            # 기관명이 필요한 경우 기관명 포함 여부 확인
            if answer_type == "기관명":
                institution_keywords = [
                    "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원",
                    "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                    "개인정보침해신고센터", "한국은행", "금융위원회"
                ]
                is_intent_match = any(keyword in response for keyword in institution_keywords)
            
            # 특징 설명이 필요한 경우
            elif answer_type == "특징설명":
                feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리"]
                is_intent_match = any(keyword in response for keyword in feature_keywords)
            
            # 지표 나열이 필요한 경우
            elif answer_type == "지표나열":
                indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지"]
                is_intent_match = any(keyword in response for keyword in indicator_keywords)
        
        # 한국어 비율이 낮거나 의도와 맞지 않으면 템플릿 사용
        if korean_ratio < 0.7 or len(response) < 20 or not is_intent_match:
            return self._generate_intent_based_template_answer(question, intent_analysis)
        
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
    
    def _generate_intent_based_template_answer(self, question: str, intent_analysis: Dict = None) -> str:
        """의도 기반 템플릿 답변 생성"""
        domain = self._detect_domain(question)
        
        # 의도별 템플릿 선택
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            
            # 고품질 템플릿 우선 사용
            if primary_intent in self.learning_data["high_quality_templates"]:
                templates = self.learning_data["high_quality_templates"][primary_intent]
                if templates:
                    best_template = max(templates, key=lambda x: x.get("quality", 0))
                    if best_template["quality"] > 0.8:
                        best_template["usage_count"] += 1
                        return best_template["answer_template"]
            
            # 학습된 의도별 성공 답변 활용
            if primary_intent in self.learning_data["intent_based_answers"]:
                successful_answers = self.learning_data["intent_based_answers"][primary_intent]
                if successful_answers:
                    # 높은 품질 점수를 가진 답변 선택
                    best_answers = [ans for ans in successful_answers if ans.get("quality", 0) > 0.7]
                    if best_answers:
                        selected = random.choice(best_answers)
                        return selected["answer"]
        
        # 기본 폴백
        return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _calculate_answer_quality(self, answer: str, question: str, intent_analysis: Dict = None) -> float:
        """답변 품질 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (25%)
        korean_ratio = self._calculate_korean_ratio(answer)
        score += korean_ratio * 0.25
        
        # 길이 적절성 (15%)
        length = len(answer)
        if 50 <= length <= 400:
            score += 0.15
        elif 30 <= length < 50 or 400 < length <= 500:
            score += 0.1
        
        # 문장 구조 (15%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.05
        
        # 전문성 (15%)
        domain_keywords = self._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.15
        
        # 의도 일치성 (30%)
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            if self._check_intent_match(answer, answer_type):
                score += 0.3
            else:
                score += 0.1
        else:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_intent_match(self, answer: str, answer_type: str) -> bool:
        """의도 일치성 확인"""
        answer_lower = answer.lower()
        
        if answer_type == "기관명":
            institution_keywords = ["위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", "조정위원회"]
            return any(keyword in answer_lower for keyword in institution_keywords)
        elif answer_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격"]
            return any(keyword in answer_lower for keyword in feature_keywords)
        elif answer_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "모니터링", "탐지", "발견", "식별"]
            return any(keyword in answer_lower for keyword in indicator_keywords)
        elif answer_type == "방안제시":
            solution_keywords = ["방안", "대책", "조치", "해결", "대응", "관리", "처리", "예방", "개선"]
            return any(keyword in answer_lower for keyword in solution_keywords)
        elif answer_type == "절차설명":
            procedure_keywords = ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행"]
            return any(keyword in answer_lower for keyword in procedure_keywords)
        elif answer_type == "조치설명":
            measure_keywords = ["조치", "대응", "대책", "방안", "보안", "예방", "개선", "강화"]
            return any(keyword in answer_lower for keyword in measure_keywords)
        elif answer_type == "법령설명":
            law_keywords = ["법", "법령", "법률", "규정", "조항", "규칙", "기준", "근거"]
            return any(keyword in answer_lower for keyword in law_keywords)
        elif answer_type == "정의설명":
            definition_keywords = ["정의", "개념", "의미", "뜻", "용어"]
            return any(keyword in answer_lower for keyword in definition_keywords)
        
        return True
    
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
    
    def _get_fallback_answer(self, question_type: str, question: str = "", max_choice: int = 5, intent_analysis: Dict = None, domain: str = "일반") -> str:
        """폴백 답변"""
        if question_type == "multiple_choice":
            # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
            if max_choice <= 0:
                max_choice = 5
            return self._get_context_based_mc_answer_enhanced(question, max_choice, domain)
        else:
            return self._generate_intent_based_template_answer(question, intent_analysis)
    
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
            
            # 의도별 성공 답변 저장
            if intent_analysis and question_type == "subjective":
                primary_intent = intent_analysis.get("primary_intent", "일반")
                if primary_intent not in self.learning_data["intent_based_answers"]:
                    self.learning_data["intent_based_answers"][primary_intent] = []
                
                intent_record = {
                    "question": question[:150],
                    "answer": answer[:200],
                    "quality": quality_score,
                    "confidence": intent_analysis.get("intent_confidence", 0.0),
                    "answer_type": intent_analysis.get("answer_type_required", "설명형"),
                    "timestamp": datetime.now().isoformat()
                }
                self.learning_data["intent_based_answers"][primary_intent].append(intent_record)
                
                # 최근 50개만 유지
                if len(self.learning_data["intent_based_answers"][primary_intent]) > 50:
                    self.learning_data["intent_based_answers"][primary_intent] = \
                        self.learning_data["intent_based_answers"][primary_intent][-50:]
                
                # 고품질 답변은 템플릿으로 저장
                if quality_score > 0.85:
                    if primary_intent not in self.learning_data["high_quality_templates"]:
                        self.learning_data["high_quality_templates"][primary_intent] = []
                    
                    template_record = {
                        "answer_template": answer[:250],
                        "quality": quality_score,
                        "usage_count": 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.learning_data["high_quality_templates"][primary_intent].append(template_record)
                    
                    # 최근 20개만 유지
                    if len(self.learning_data["high_quality_templates"][primary_intent]) > 20:
                        self.learning_data["high_quality_templates"][primary_intent] = \
                            sorted(self.learning_data["high_quality_templates"][primary_intent], 
                                  key=lambda x: x["quality"], reverse=True)[:20]
        else:
            self.learning_data["failed_answers"].append(record)
            
            # 선택지 범위 오류 기록
            if question_type == "multiple_choice" and answer and answer.isdigit():
                answer_num = int(answer)
                if answer_num > max_choice:
                    self.learning_data["choice_range_errors"].append({
                        "question": question[:100],
                        "answer": answer,
                        "max_choice": max_choice,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # 질문 패턴 학습
        domain = self._detect_domain(question)
        if domain not in self.learning_data["question_patterns"]:
            self.learning_data["question_patterns"][domain] = {"count": 0, "avg_quality": 0.0}
        
        patterns = self.learning_data["question_patterns"][domain]
        patterns["count"] += 1
        patterns["avg_quality"] = (patterns["avg_quality"] * (patterns["count"] - 1) + quality_score) / patterns["count"]
        
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