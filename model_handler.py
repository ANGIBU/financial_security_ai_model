# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 학습 데이터 저장
- 질문 의도 기반 답변 생성
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
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG,
    GENERATION_CONFIG,
    OPTIMIZATION_CONFIG,
    PKL_DIR,
    JSON_CONFIG_FILES,
    MEMORY_CONFIG,
    get_device,
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
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
            use_fast=MODEL_CONFIG["use_fast_tokenizer"],
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
            device_map=MODEL_CONFIG["device_map"],
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
        )

        self.model.eval()

        # 워밍업
        self._warmup()

        if verbose:
            print("모델 로딩 완료")

        # 학습 데이터 로드 현황
        if len(self.learning_data["successful_answers"]) > 0 and verbose:
            print(
                f"이전 학습 데이터 로드: 성공 {len(self.learning_data['successful_answers'])}개, 실패 {len(self.learning_data['failed_answers'])}개"
            )

    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            # model_config.json 로드
            with open(JSON_CONFIG_FILES["model_config"], "r", encoding="utf-8") as f:
                model_config = json.load(f)

            # 모델 관련 데이터 할당
            self.mc_context_patterns = model_config["mc_context_patterns"]
            self.intent_specific_prompts = model_config["intent_specific_prompts"]
            self.answer_distributions = model_config[
                "answer_distribution_default"
            ].copy()
            self.mc_answer_counts = model_config["mc_answer_counts_default"].copy()
            self.learning_data_structure = model_config["learning_data_structure"]

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
            "domain_specific_patterns": {},
        }

        self.intent_specific_prompts = {
            "기관_묻기": ["다음 질문에서 요구하는 특정 기관명을 정확히 답변하세요."],
            "특징_묻기": [
                "해당 항목의 핵심적인 특징들을 구체적으로 나열하고 설명하세요."
            ],
            "지표_묻기": [
                "탐지 지표와 징후를 중심으로 구체적으로 나열하고 설명하세요."
            ],
            "방안_묻기": ["구체적인 대응 방안과 해결책을 제시하세요."],
            "절차_묻기": ["단계별 절차를 순서대로 설명하세요."],
            "조치_묻기": ["필요한 보안조치와 대응조치를 설명하세요."],
        }

        self.answer_distributions = {
            3: {"1": 0, "2": 0, "3": 0},
            4: {"1": 0, "2": 0, "3": 0, "4": 0},
            5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
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
        }

    def _load_learning_data(self):
        """이전 학습 데이터 로드"""
        learning_file = self.pkl_dir / "learning_data.pkl"

        if learning_file.exists():
            try:
                with open(learning_file, "rb") as f:
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
                "successful_answers": self.learning_data["successful_answers"][
                    -MEMORY_CONFIG["max_learning_records"]["successful_answers"] :
                ],
                "failed_answers": self.learning_data["failed_answers"][
                    -MEMORY_CONFIG["max_learning_records"]["failed_answers"] :
                ],
                "question_patterns": self.learning_data["question_patterns"],
                "answer_quality_scores": self.learning_data["answer_quality_scores"][
                    -MEMORY_CONFIG["max_learning_records"]["quality_scores"] :
                ],
                "mc_context_patterns": self.learning_data["mc_context_patterns"],
                "choice_range_errors": self.learning_data["choice_range_errors"][
                    -MEMORY_CONFIG["max_learning_records"]["choice_range_errors"] :
                ],
                "intent_based_answers": self.learning_data["intent_based_answers"],
                "domain_specific_learning": self.learning_data[
                    "domain_specific_learning"
                ],
                "intent_prompt_effectiveness": self.learning_data[
                    "intent_prompt_effectiveness"
                ],
                "high_quality_templates": self.learning_data["high_quality_templates"],
                "mc_accuracy_by_domain": self.learning_data["mc_accuracy_by_domain"],
                "negative_vs_positive_patterns": self.learning_data[
                    "negative_vs_positive_patterns"
                ],
                "choice_distribution_learning": self.learning_data[
                    "choice_distribution_learning"
                ],
                "last_updated": datetime.now().isoformat(),
            }

            with open(learning_file, "wb") as f:
                pickle.dump(save_data, f)

        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")

    def _extract_choice_count(self, question: str) -> int:
        """질문에서 선택지 개수 추출"""
        lines = question.split("\n")
        choice_numbers = []

        for line in lines:
            # 선택지 패턴: 숫자 + 공백 + 내용
            match = re.match(r"^(\d+)\s+(.+)", line.strip())
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
            pattern = r"1\s.*" + ".*".join([f"{j}\s" for j in range(2, i + 1)])
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
            "confidence_score": 0.0,
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
            keyword_matches = sum(
                1 for keyword in domain_info["keywords"] if keyword in question_lower
            )

            if keyword_matches > 0:
                context["domain_hints"].append(domain)
                context["likely_answers"] = domain_info["common_answers"]
                context["confidence_score"] = min(
                    keyword_matches / len(domain_info["keywords"]), 1.0
                )

        # 핵심 용어 추출
        domain_terms = {
            "금융투자": ["구분", "업무", "금융투자업", "해당하지"],
            "위험관리": ["요소", "계획", "위험", "적절하지"],
            "개인정보보호": ["정책", "수립", "요소", "중요한"],
            "전자금융": ["요구", "경우", "자료제출", "통화신용정책"],
            "사이버보안": ["활용", "이유", "SBOM", "소프트웨어"],
            "정보보안": ["복구", "계획", "절차", "옳지"],
        }

        if domain in domain_terms:
            for term in domain_terms[domain]:
                if term in question:
                    context["key_terms"].append(term)

        return context

    def _create_intent_aware_prompt(
        self, question: str, intent_analysis: Dict, domain_hints: Dict = None
    ) -> str:
        """의도 인식 기반 프롬프트 생성"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        domain = self._detect_domain(question)
        context_hints = intent_analysis.get("context_hints", [])
        intent_confidence = intent_analysis.get("intent_confidence", 0.0)

        # 도메인 힌트 정보 추가
        hint_context = ""
        if domain_hints:
            if "template_hints" in domain_hints and domain_hints["template_hints"]:
                hint_context += f"\n참고 정보: {domain_hints['template_hints']}"

            if (
                "institution_hints" in domain_hints
                and domain_hints["institution_hints"]
            ):
                hint_context += f"\n기관 관련 정보: {domain_hints['institution_hints']}"

            if "improvement_type" in domain_hints:
                improvement_type = domain_hints["improvement_type"]
                if improvement_type == "korean_ratio_low":
                    hint_context += "\n답변 작성 시 한국어만 사용하여 작성하세요."
                elif improvement_type == "intent_mismatch":
                    hint_context += f"\n질문의 의도({primary_intent})에 정확히 부합하는 답변을 작성하세요."
                elif improvement_type == "quality_low":
                    hint_context += (
                        "\n전문적이고 구체적인 내용으로 답변 품질을 높여주세요."
                    )

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
            context_instruction = (
                f"답변 시 다음 사항을 고려하세요: {', '.join(context_hints)}"
            )

        # 고품질 프롬프트 패턴 활용
        if primary_intent in self.learning_data["high_quality_templates"]:
            templates = self.learning_data["high_quality_templates"][primary_intent]
            if templates:
                best_template = max(templates, key=lambda x: x.get("quality", 0))
                if best_template["quality"] > 0.8:
                    if "answer_template" in best_template:
                        # 템플릿을 힌트로 활용하여 새로운 프롬프트 생성
                        hint_context += f"\n참고 답변 패턴: {best_template['answer_template'][:100]}..."

        prompts = [
            f"""금융보안 전문가로서 다음 {domain} 관련 질문에 한국어로만 정확한 답변을 작성하세요.

질문: {question}

{intent_instruction}
{type_guidance}
{context_instruction}
{hint_context}

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
{hint_context}

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
{hint_context}

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
{hint_context}

답변 원칙:
- 한국어 전용 작성
- 의도에 정확히 부합
- 구체적이고 실무적인 내용
- 관련 법령 근거 포함

답변:""",
        ]

        # 프롬프트 효과성 기록
        selected_prompt = random.choice(prompts)
        prompt_id = hash(selected_prompt) % 1000

        if primary_intent not in self.learning_data["intent_prompt_effectiveness"]:
            self.learning_data["intent_prompt_effectiveness"][primary_intent] = {}

        if (
            prompt_id
            not in self.learning_data["intent_prompt_effectiveness"][primary_intent]
        ):
            self.learning_data["intent_prompt_effectiveness"][primary_intent][
                prompt_id
            ] = {
                "prompt": selected_prompt,
                "use_count": 0,
                "success_count": 0,
                "avg_quality": 0.0,
            }

        self.learning_data["intent_prompt_effectiveness"][primary_intent][prompt_id][
            "use_count"
        ] += 1

        return selected_prompt

    def generate_answer(
        self,
        question: str,
        question_type: str,
        max_choice: int = 5,
        intent_analysis: Dict = None,
        domain_hints: Dict = None,
    ) -> str:
        """답변 생성 - 반드시 LLM 사용"""

        # 도메인 감지
        domain = self._detect_domain(question)

        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_enhanced_mc_prompt(
                question, max_choice, domain, domain_hints
            )
        else:
            if intent_analysis:
                prompt = self._create_intent_aware_prompt(
                    question, intent_analysis, domain_hints
                )
            else:
                prompt = self._create_korean_subj_prompt(question, domain, domain_hints)

        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1500
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            # 생성 설정
            gen_config = self._get_generation_config(question_type)

            # 모델 실행
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)

            # 디코딩
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            # 후처리
            if question_type == "multiple_choice":
                answer = self._process_enhanced_mc_answer(
                    response, question, max_choice, domain
                )
                # 선택지 범위 검증
                if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                    self._add_learning_record(
                        question,
                        answer,
                        question_type,
                        True,
                        max_choice,
                        1.0,
                        intent_analysis,
                    )

                    # 도메인별 정확도 업데이트
                    if domain in self.learning_data["mc_accuracy_by_domain"]:
                        self.learning_data["mc_accuracy_by_domain"][domain][
                            "correct"
                        ] += 1

                    return answer
                else:
                    # 범위 벗어난 경우 재시도 없이 범위 내 답변 강제 생성
                    valid_answer = self._force_valid_mc_answer(response, max_choice)
                    self._add_learning_record(
                        question,
                        valid_answer,
                        question_type,
                        False,
                        max_choice,
                        0.5,
                        intent_analysis,
                    )
                    return valid_answer
            else:
                answer = self._process_intent_aware_subj_answer(
                    response, question, intent_analysis
                )
                korean_ratio = self._calculate_korean_ratio(answer)
                quality_score = self._calculate_answer_quality(
                    answer, question, intent_analysis
                )
                success = korean_ratio > 0.7 and quality_score > 0.5

                self._add_learning_record(
                    question,
                    answer,
                    question_type,
                    success,
                    max_choice,
                    quality_score,
                    intent_analysis,
                )
                return answer

        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            fallback = self._get_fallback_answer_with_llm(
                question_type, question, max_choice, intent_analysis, domain
            )
            self._add_learning_record(
                question,
                fallback,
                question_type,
                False,
                max_choice,
                0.3,
                intent_analysis,
            )
            return fallback

    def generate_contextual_mc_answer(
        self, question: str, max_choice: int, domain: str
    ) -> str:
        """컨텍스트 기반 객관식 답변 생성 - LLM 사용"""
        context_hints = self._analyze_mc_context(question, domain)

        # 컨텍스트 정보를 포함한 특수 프롬프트
        prompt = self._create_contextual_mc_prompt(
            question, max_choice, domain, context_hints
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1000
            )
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            gen_config = self._get_generation_config("multiple_choice")

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            answer = self._process_enhanced_mc_answer(
                response, question, max_choice, domain
            )

            if not (answer and answer.isdigit() and 1 <= int(answer) <= max_choice):
                answer = self._force_valid_mc_answer(response, max_choice)

            return answer

        except Exception as e:
            if self.verbose:
                print(f"컨텍스트 기반 답변 생성 오류: {e}")
            return self._force_valid_mc_answer("", max_choice)

    def generate_improved_answer(
        self,
        question: str,
        question_type: str,
        max_choice: int,
        intent_analysis: Dict,
        improvement_hints: Dict,
    ) -> str:
        """개선된 답변 생성 - LLM 사용"""
        return self.generate_answer(
            question, question_type, max_choice, intent_analysis, improvement_hints
        )

    def generate_fallback_mc_answer(
        self, question: str, max_choice: int, domain: str
    ) -> str:
        """폴백 객관식 답변 생성 - LLM 사용"""
        return self.generate_contextual_mc_answer(question, max_choice, domain)

    def generate_fallback_subjective_answer(self, question: str) -> str:
        """폴백 주관식 답변 생성 - LLM 사용"""
        domain = self._detect_domain(question)
        prompt = self._create_korean_subj_prompt(
            question, domain, {"fallback_mode": True}
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1000
            )
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            gen_config = self._get_generation_config("subjective")

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            return self._process_intent_aware_subj_answer(response, question, None)

        except Exception as e:
            if self.verbose:
                print(f"폴백 주관식 답변 생성 오류: {e}")
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."

    def _create_enhanced_mc_prompt(
        self,
        question: str,
        max_choice: int,
        domain: str = "일반",
        domain_hints: Dict = None,
    ) -> str:
        """객관식 프롬프트 생성"""
        if max_choice <= 0:
            max_choice = 5

        context = self._analyze_mc_context(question, domain)

        # 선택지 범위 명시
        choice_range = (
            "에서 ".join([str(i) for i in range(1, max_choice + 1)]) + f"번 중"
        )

        # 힌트 정보 추가
        hint_context = ""
        if (
            domain_hints
            and "pattern_hints" in domain_hints
            and domain_hints["pattern_hints"]
        ):
            hint_context = f"\n참고 정보: {domain_hints['pattern_hints']}"

        if domain_hints and "context_hints" in domain_hints:
            context_info = domain_hints["context_hints"]
            if context_info.get("is_negative"):
                hint_context += "\n이 문제는 '해당하지 않는' 또는 '적절하지 않은' 것을 찾는 문제입니다."
            elif context_info.get("is_positive"):
                hint_context += "\n이 문제는 '적절한' 또는 '옳은' 것을 찾는 문제입니다."

        # 컨텍스트와 도메인에 따른 프롬프트 조정
        if context["is_negative"]:
            if domain == "금융투자":
                instruction = f"금융투자업의 구분에서 해당하지 않는 것을 {choice_range} 찾으세요. 소비자금융업, 투자자문업, 투자매매업, 투자중개업, 보험중개업을 신중히 검토하세요."
            elif domain == "위험관리":
                instruction = f"위험 관리 계획 수립 시 고려해야 할 요소 중 적절하지 않은 것을 {choice_range} 찾으세요."
            elif domain == "정보보안":
                instruction = f"재해 복구 계획 수립 시 고려해야 할 요소 중 옳지 않은 것을 {choice_range} 찾으세요."
            else:
                instruction = (
                    f"다음 중 해당하지 않거나 옳지 않은 것을 {choice_range} 찾으세요."
                )
        elif context["is_positive"]:
            if domain == "개인정보보호":
                instruction = (
                    f"정책 수립 단계에서 가장 중요한 요소를 {choice_range} 찾으세요."
                )
            elif domain == "전자금융":
                instruction = f"한국은행이 자료제출을 요구할 수 있는 경우를 {choice_range} 찾으세요."
            elif domain == "사이버보안":
                instruction = (
                    f"SBOM을 활용하는 가장 적절한 이유를 {choice_range} 찾으세요."
                )
            else:
                instruction = (
                    f"다음 중 가장 적절하거나 옳은 것을 {choice_range} 찾으세요."
                )
        else:
            instruction = f"정답을 {choice_range} 선택하세요."

        prompts = [
            f"""다음은 {domain} 분야의 금융보안 관련 문제입니다. {instruction}

{question}
{hint_context}

위 문제를 신중히 분석하고, 1부터 {max_choice}까지 중 하나의 정답을 선택하세요.
각 선택지를 꼼꼼히 검토한 후 정답 번호만 답하세요.

정답:""",
            f"""금융보안 전문가로서 다음 {domain} 문제를 해결하세요.

{question}
{hint_context}

{instruction}
선택지를 모두 검토한 후 1부터 {max_choice}번 중 정답을 선택하세요.
번호만 답하세요.

답:""",
            f"""다음 {domain} 분야 금융보안 문제를 분석하고 정답을 선택하세요.

문제: {question}
{hint_context}

{instruction}
정답을 1부터 {max_choice}번 중 하나의 번호로만 답하세요.

정답:""",
        ]

        return random.choice(prompts)

    def _create_contextual_mc_prompt(
        self, question: str, max_choice: int, domain: str, context_hints: Dict
    ) -> str:
        """컨텍스트 기반 객관식 프롬프트"""
        choice_range = (
            "에서 ".join([str(i) for i in range(1, max_choice + 1)]) + f"번 중"
        )

        context_info = ""
        if context_hints.get("is_negative"):
            context_info = (
                "이 문제는 해당하지 않거나 적절하지 않은 것을 찾는 문제입니다."
            )
        elif context_hints.get("is_positive"):
            context_info = "이 문제는 가장 적절하거나 옳은 것을 찾는 문제입니다."

        if context_hints.get("key_terms"):
            context_info += f" 핵심 용어: {', '.join(context_hints['key_terms'])}"

        return f"""다음은 {domain} 분야의 객관식 문제입니다.

{question}

{context_info}

문제를 신중히 분석하여 {choice_range} 정답을 선택하세요.
반드시 번호만 답하세요.

정답:"""

    def _create_korean_subj_prompt(
        self, question: str, domain: str = "일반", domain_hints: Dict = None
    ) -> str:
        """한국어 전용 주관식 프롬프트 생성"""

        # 힌트 정보 추가
        hint_context = ""
        if domain_hints:
            if "template_hints" in domain_hints and domain_hints["template_hints"]:
                hint_context += f"\n참고 정보: {domain_hints['template_hints']}"

            if "fallback_mode" in domain_hints:
                hint_context += "\n기본적인 답변을 한국어로만 작성하세요."

        prompts = [
            f"""금융보안 전문가로서 다음 {domain} 분야 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}
{hint_context}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명
- 영어 용어 사용 금지

답변:""",
            f"""다음은 {domain} 분야의 전문 질문입니다. 한국어로만 상세하고 정확한 답변을 제공하세요.

{question}
{hint_context}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 법적 근거와 실무 절차를 한국어로 설명
- 영어나 외국어 사용 금지

답변:""",
            f"""{domain} 전문가의 관점에서 다음 질문에 한국어로만 답변하세요.

질문: {question}
{hint_context}

답변 요구사항:
- 완전한 한국어 답변
- 관련 법령과 규정을 한국어로 설명
- 체계적이고 구체적인 한국어 설명

답변:""",
        ]

        return random.choice(prompts)

    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정"""
        config_dict = GENERATION_CONFIG[question_type].copy()
        config_dict["pad_token_id"] = self.tokenizer.pad_token_id
        config_dict["eos_token_id"] = self.tokenizer.eos_token_id

        return GenerationConfig(**config_dict)

    def _process_enhanced_mc_answer(
        self, response: str, question: str, max_choice: int, domain: str = "일반"
    ) -> str:
        """객관식 답변 처리"""
        if max_choice <= 0:
            max_choice = 5

        # 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r"[1-9]", response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] += 1
                    self.mc_answer_counts[max_choice] += 1
                return num

        # 유효한 답변을 찾지 못한 경우 강제 생성
        return self._force_valid_mc_answer(response, max_choice)

    def _force_valid_mc_answer(self, response: str, max_choice: int) -> str:
        """유효한 객관식 답변 강제 생성"""
        if max_choice <= 0:
            max_choice = 5

        # 응답에서 숫자 패턴 분석
        all_numbers = re.findall(r"\d+", response)

        # 가장 적절한 숫자 선택
        for num_str in all_numbers:
            num = int(num_str)
            if 1 <= num <= max_choice:
                return str(num)

        # 마지막 수단: 중간값 선택
        return str((max_choice + 1) // 2)

    def _process_intent_aware_subj_answer(
        self, response: str, question: str, intent_analysis: Dict = None
    ) -> str:
        """의도 인식 기반 주관식 답변 처리"""
        # 기본 정리
        response = re.sub(r"\s+", " ", response).strip()

        # 잘못된 인코딩으로 인한 깨진 문자 제거
        response = re.sub(r"[^\w\s가-힣.,!?()[\]\-]", " ", response)
        response = re.sub(r"\s+", " ", response).strip()

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
                    "위원회",
                    "감독원",
                    "은행",
                    "기관",
                    "센터",
                    "청",
                    "부",
                    "원",
                    "전자금융분쟁조정위원회",
                    "금융감독원",
                    "개인정보보호위원회",
                    "개인정보침해신고센터",
                    "한국은행",
                    "금융위원회",
                ]
                is_intent_match = any(
                    keyword in response for keyword in institution_keywords
                )

            # 특징 설명이 필요한 경우
            elif answer_type == "특징설명":
                feature_keywords = [
                    "특징",
                    "특성",
                    "속성",
                    "성질",
                    "기능",
                    "역할",
                    "원리",
                ]
                is_intent_match = any(
                    keyword in response for keyword in feature_keywords
                )

            # 지표 나열이 필요한 경우
            elif answer_type == "지표나열":
                indicator_keywords = [
                    "지표",
                    "신호",
                    "징후",
                    "패턴",
                    "행동",
                    "활동",
                    "모니터링",
                    "탐지",
                ]
                is_intent_match = any(
                    keyword in response for keyword in indicator_keywords
                )

        # 품질이 너무 낮으면 기본 응답 사용
        if korean_ratio < 0.5 or len(response) < 20 or not is_intent_match:
            if intent_analysis and intent_analysis.get("primary_intent"):
                return self._generate_basic_intent_answer(
                    intent_analysis["primary_intent"]
                )
            else:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."

        # 길이 제한
        if len(response) > 350:
            sentences = response.split(". ")
            response = ". ".join(sentences[:3])
            if not response.endswith("."):
                response += "."

        # 마침표 확인
        if not response.endswith((".", "다", "요", "함")):
            response += "."

        return response

    def _generate_basic_intent_answer(self, primary_intent: str) -> str:
        """기본 의도별 답변 생성"""
        intent_responses = {
            "기관_묻기": "해당 분야의 전문 기관에서 관련 업무를 담당하고 있습니다.",
            "특징_묻기": "주요 특징과 성질을 체계적으로 분석하고 관리해야 합니다.",
            "지표_묻기": "관련 지표와 징후를 지속적으로 모니터링하고 분석해야 합니다.",
            "방안_묻기": "체계적인 관리 방안을 수립하고 지속적인 개선활동을 수행해야 합니다.",
            "절차_묻기": "관련 법령과 규정에 따라 절차를 수립하고 체계적으로 관리해야 합니다.",
            "조치_묣기": "적절한 보안조치와 대응조치를 마련하여 체계적으로 관리해야 합니다.",
        }

        return intent_responses.get(
            primary_intent,
            "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
        )

    def _calculate_answer_quality(
        self, answer: str, question: str, intent_analysis: Dict = None
    ) -> float:
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
        if answer.endswith((".", "다", "요", "함")):
            score += 0.1

        sentences = answer.split(".")
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
            institution_keywords = [
                "위원회",
                "감독원",
                "은행",
                "기관",
                "센터",
                "청",
                "부",
                "원",
                "조정위원회",
            ]
            return any(keyword in answer_lower for keyword in institution_keywords)
        elif answer_type == "특징설명":
            feature_keywords = [
                "특징",
                "특성",
                "속성",
                "성질",
                "기능",
                "역할",
                "원리",
                "성격",
            ]
            return any(keyword in answer_lower for keyword in feature_keywords)
        elif answer_type == "지표나열":
            indicator_keywords = [
                "지표",
                "신호",
                "징후",
                "패턴",
                "행동",
                "모니터링",
                "탐지",
                "발견",
                "식별",
            ]
            return any(keyword in answer_lower for keyword in indicator_keywords)
        elif answer_type == "방안제시":
            solution_keywords = [
                "방안",
                "대책",
                "조치",
                "해결",
                "대응",
                "관리",
                "처리",
                "예방",
                "개선",
            ]
            return any(keyword in answer_lower for keyword in solution_keywords)
        elif answer_type == "절차설명":
            procedure_keywords = [
                "절차",
                "과정",
                "단계",
                "순서",
                "프로세스",
                "진행",
                "수행",
            ]
            return any(keyword in answer_lower for keyword in procedure_keywords)
        elif answer_type == "조치설명":
            measure_keywords = [
                "조치",
                "대응",
                "대책",
                "방안",
                "보안",
                "예방",
                "개선",
                "강화",
            ]
            return any(keyword in answer_lower for keyword in measure_keywords)
        elif answer_type == "법령설명":
            law_keywords = [
                "법",
                "법령",
                "법률",
                "규정",
                "조항",
                "규칙",
                "기준",
                "근거",
            ]
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

    def _get_fallback_answer_with_llm(
        self,
        question_type: str,
        question: str = "",
        max_choice: int = 5,
        intent_analysis: Dict = None,
        domain: str = "일반",
    ) -> str:
        """폴백 답변 - LLM 사용"""
        if question_type == "multiple_choice":
            if max_choice <= 0:
                max_choice = 5
            return self.generate_fallback_mc_answer(question, max_choice, domain)
        else:
            return self.generate_fallback_subjective_answer(question)

    def _detect_domain(self, question: str) -> str:
        """도메인 감지"""
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in ["개인정보", "정보주체", "만 14세", "법정대리인"]
        ):
            return "개인정보보호"
        elif any(
            word in question_lower
            for word in ["트로이", "악성코드", "RAT", "원격제어", "딥페이크", "SBOM"]
        ):
            return "사이버보안"
        elif any(
            word in question_lower
            for word in ["전자금융", "전자적", "분쟁조정", "금융감독원"]
        ):
            return "전자금융"
        elif any(
            word in question_lower
            for word in ["정보보안", "isms", "관리체계", "정책 수립"]
        ):
            return "정보보안"
        elif any(
            word in question_lower
            for word in ["위험관리", "위험 관리", "재해복구", "위험수용"]
        ):
            return "위험관리"
        elif any(
            word in question_lower for word in ["금융투자", "투자자문", "금융투자업"]
        ):
            return "금융투자"
        else:
            return "일반"

    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0

        korean_chars = len(re.findall(r"[가-힣]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))

        if total_chars == 0:
            return 0.0

        return korean_chars / total_chars

    def _add_learning_record(
        self,
        question: str,
        answer: str,
        question_type: str,
        success: bool,
        max_choice: int = 5,
        quality_score: float = 0.0,
        intent_analysis: Dict = None,
    ):
        """학습 기록 추가"""
        record = {
            "question": question[:200],
            "answer": answer[:300],
            "type": question_type,
            "max_choice": max_choice,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
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
                    "answer_type": intent_analysis.get(
                        "answer_type_required", "설명형"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
                self.learning_data["intent_based_answers"][primary_intent].append(
                    intent_record
                )

                # 최근 50개만 유지
                if len(self.learning_data["intent_based_answers"][primary_intent]) > 50:
                    self.learning_data["intent_based_answers"][primary_intent] = (
                        self.learning_data["intent_based_answers"][primary_intent][-50:]
                    )

                # 고품질 답변은 템플릿으로 저장
                if quality_score > 0.85:
                    if (
                        primary_intent
                        not in self.learning_data["high_quality_templates"]
                    ):
                        self.learning_data["high_quality_templates"][
                            primary_intent
                        ] = []

                    template_record = {
                        "answer_template": answer[:250],
                        "quality": quality_score,
                        "usage_count": 0,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.learning_data["high_quality_templates"][primary_intent].append(
                        template_record
                    )

                    # 최근 20개만 유지
                    if (
                        len(
                            self.learning_data["high_quality_templates"][primary_intent]
                        )
                        > 20
                    ):
                        self.learning_data["high_quality_templates"][primary_intent] = (
                            sorted(
                                self.learning_data["high_quality_templates"][
                                    primary_intent
                                ],
                                key=lambda x: x["quality"],
                                reverse=True,
                            )[:20]
                        )
        else:
            self.learning_data["failed_answers"].append(record)

            # 선택지 범위 오류 기록
            if question_type == "multiple_choice" and answer and answer.isdigit():
                answer_num = int(answer)
                if answer_num > max_choice:
                    self.learning_data["choice_range_errors"].append(
                        {
                            "question": question[:100],
                            "answer": answer,
                            "max_choice": max_choice,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # 질문 패턴 학습
        domain = self._detect_domain(question)
        if domain not in self.learning_data["question_patterns"]:
            self.learning_data["question_patterns"][domain] = {
                "count": 0,
                "avg_quality": 0.0,
            }

        patterns = self.learning_data["question_patterns"][domain]
        patterns["count"] += 1
        patterns["avg_quality"] = (
            patterns["avg_quality"] * (patterns["count"] - 1) + quality_score
        ) / patterns["count"]

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
                _ = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
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
        }

    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        return {
            "successful_count": len(self.learning_data["successful_answers"]),
            "failed_count": len(self.learning_data["failed_answers"]),
            "choice_range_errors": len(self.learning_data["choice_range_errors"]),
            "question_patterns": dict(self.learning_data["question_patterns"]),
            "intent_based_answers_count": {
                k: len(v) for k, v in self.learning_data["intent_based_answers"].items()
            },
            "high_quality_templates_count": {
                k: len(v)
                for k, v in self.learning_data["high_quality_templates"].items()
            },
            "mc_accuracy_by_domain": dict(self.learning_data["mc_accuracy_by_domain"]),
            "avg_quality": (
                sum(self.learning_data["answer_quality_scores"])
                / len(self.learning_data["answer_quality_scores"])
                if self.learning_data["answer_quality_scores"]
                else 0
            ),
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            # 학습 데이터 저장
            self._save_learning_data()

            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")
