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
import unicodedata
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

        # 토크나이저 로드 (한국어 최적화)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
            use_fast=MODEL_CONFIG["use_fast_tokenizer"],
        )

        # 한국어 처리 최적화
        self._optimize_tokenizer_for_korean()

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

    def _optimize_tokenizer_for_korean(self):
        """토크나이저 한국어 최적화"""
        # 한국어 처리 개선을 위한 설정
        if hasattr(self.tokenizer, "do_lower_case"):
            self.tokenizer.do_lower_case = False

        # 한국어 정규화 비활성화 (정보 손실 방지)
        if hasattr(self.tokenizer, "normalize"):
            self.tokenizer.normalize = False

        # 특수 토큰 추가 (필요시)
        special_tokens = ["<korean>", "</korean>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            # model_config.json 로드
            with open(JSON_CONFIG_FILES["model_config"], "r", encoding="utf-8") as f:
                model_config = json.load(f)

            # processing_config.json 로드 (한국어 복구 설정 포함)
            with open(
                JSON_CONFIG_FILES["processing_config"], "r", encoding="utf-8"
            ) as f:
                processing_config = json.load(f)

            # 모델 관련 데이터 할당
            self.mc_context_patterns = model_config["mc_context_patterns"]
            self.intent_specific_prompts = model_config["intent_specific_prompts"]
            self.answer_distributions = model_config[
                "answer_distribution_default"
            ].copy()
            self.mc_answer_counts = model_config["mc_answer_counts_default"].copy()
            self.learning_data_structure = model_config["learning_data_structure"]

            # 한국어 복구 설정 로드
            self.korean_recovery_config = processing_config["korean_text_recovery"]
            self.korean_quality_patterns = processing_config["korean_quality_patterns"]

            # 한국어 복구 매핑 구성
            self._setup_korean_recovery_mappings()

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

    def _setup_korean_recovery_mappings(self):
        """JSON에서 로드한 한국어 복구 매핑 설정"""
        # 복구 매핑 통합
        self.korean_recovery_mapping = {}

        # 깨진 유니코드 문자 제거
        for broken, replacement in self.korean_recovery_config[
            "broken_unicode_chars"
        ].items():
            # 유니코드 이스케이프 시퀀스를 실제 문자로 변환
            try:
                actual_char = broken.encode().decode("unicode_escape")
                self.korean_recovery_mapping[actual_char] = replacement
            except:
                pass

        # 일본어 카타카나 제거
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["japanese_katakana_removal"]
        )

        # 깨진 한국어 패턴 제거
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["broken_korean_patterns"]
        )

        # 띄어쓰기 문제 수정
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["spaced_korean_fixes"]
        )

        # 일반적인 한국어 오타 수정
        self.korean_recovery_mapping.update(
            self.korean_recovery_config["common_korean_typos"]
        )

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

        # 기본 한국어 복구 매핑
        self.korean_recovery_mapping = {
            "어어지인": "",
            "선 어": "",
            "언 어": "",
            "순 어": "",
            "ᄒᆞᆫ": "",
            "작로": "으로",
            "갈취 묻는 말": "",  # 문제가 되는 패턴 제거
            "묻고 갈취": "",
            "갈취": "",
        }

        # 기본 품질 패턴
        self.korean_quality_patterns = [
            {
                "pattern": r"([가-힣])\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\s+",
                "replacement": r"\1\2 ",
            },
            {
                "pattern": r"([가-힣])\s+(다|요|함|니다|습니다)\s*\.",
                "replacement": r"\1\2.",
            },
            {"pattern": r"\s+", "replacement": " "},
        ]

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

    def detect_repetitive_patterns(self, text: str) -> bool:
        """반복 패턴 감지"""
        if not text or len(text) < 20:
            return False

        # 문제가 되는 반복 패턴들
        problematic_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취",
            r"(.{2,10})\s*\1\s*\1",  # 동일한 짧은 문구가 3번 이상 반복
            r"(.{1,5})\s*(\1\s*){5,}",  # 매우 짧은 단어가 5번 이상 반복
        ]

        for pattern in problematic_patterns:
            if re.search(pattern, text):
                return True

        # 같은 단어가 연속으로 5번 이상 나오는 경우
        words = text.split()
        if len(words) >= 5:
            for i in range(len(words) - 4):
                if (
                    words[i]
                    == words[i + 1]
                    == words[i + 2]
                    == words[i + 3]
                    == words[i + 4]
                ):
                    return True

        return False

    def remove_repetitive_patterns(self, text: str) -> str:
        """반복 패턴 제거"""
        if not text:
            return ""

        # 문제가 되는 특정 패턴 제거
        problematic_removals = [
            "갈취 묻는 말",
            "묻고 갈취",
            "갈취",
        ]

        for pattern in problematic_removals:
            text = text.replace(pattern, "")

        # 연속된 동일 단어 제거 (3개 이상)
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            # 연속된 동일 단어 개수 확인
            count = 1
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            # 최대 2개까지만 허용
            if count >= 3:
                cleaned_words.extend([current_word] * min(2, count))
            else:
                cleaned_words.extend([current_word] * count)

            i += count

        text = " ".join(cleaned_words)

        # 반복되는 구문 패턴 제거
        text = re.sub(r"(.{3,15})\s*\1\s*\1+", r"\1", text)

        # 불필요한 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def recover_korean_text(self, text: str) -> str:
        """JSON 설정 기반 한국어 텍스트 복구"""
        if not text:
            return ""

        # 1단계: 반복 패턴 제거
        if self.detect_repetitive_patterns(text):
            text = self.remove_repetitive_patterns(text)

        # 2단계: 유니코드 정규화
        text = unicodedata.normalize("NFC", text)

        # 3단계: 깨진 문자 복구 (JSON에서 로드한 매핑 사용)
        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        # 4단계: 품질 패턴 적용 (JSON에서 로드한 패턴 사용)
        for pattern_config in self.korean_quality_patterns:
            pattern = pattern_config["pattern"]
            replacement = pattern_config["replacement"]
            text = re.sub(pattern, replacement, text)

        # 5단계: 추가 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def enhance_korean_answer_quality(
        self, answer: str, question: str = "", intent_analysis: Dict = None
    ) -> str:
        """한국어 답변 품질 향상"""
        if not answer:
            return ""

        # 1단계: 반복 패턴 조기 감지 및 제거
        if self.detect_repetitive_patterns(answer):
            answer = self.remove_repetitive_patterns(answer)
            if len(answer) < 30:  # 너무 많이 제거되어 내용이 없어진 경우
                return "답변 생성 중 오류가 발생하여 재생성이 필요합니다."

        # 2단계: 기본 복구
        answer = self.recover_korean_text(answer)

        # 3단계: 의도별 개선
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            # 기관명 답변 개선
            if answer_type == "기관명":
                # 기관명 키워드 강화
                institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
                if not any(keyword in answer for keyword in institution_keywords):
                    if "전자금융" in question or "분쟁조정" in question:
                        answer = "전자금융분쟁조정위원회에서 " + answer
                    elif "개인정보" in question:
                        answer = "개인정보보호위원회에서 " + answer

            # 특징 설명 개선
            elif answer_type == "특징설명":
                if "특징" not in answer and "특성" not in answer:
                    answer = "주요 특징은 " + answer

            # 지표 나열 개선
            elif answer_type == "지표나열":
                if "지표" not in answer and "탐지" not in answer:
                    answer = "주요 탐지 지표는 " + answer

        # 4단계: 문법 및 구조 개선
        # 불완전한 문장 보완
        if len(answer) > 10 and not answer.endswith((".", "다", "요", "함")):
            if answer.endswith("니"):
                answer += "다."
            elif answer.endswith("습"):
                answer += "니다."
            else:
                answer += "."

        # 5단계: 길이 조절
        if len(answer) > 400:
            sentences = answer.split(". ")
            if len(sentences) > 3:
                answer = ". ".join(sentences[:3])
                if not answer.endswith("."):
                    answer += "."

        # 6단계: 최종 정리
        answer = re.sub(r"\s+", " ", answer).strip()

        # 7단계: 최종 검증 - 반복 패턴이 다시 생겼는지 확인
        if self.detect_repetitive_patterns(answer):
            # 반복 패턴이 다시 생긴 경우 실패 메시지로 대체
            return "생성에 실패하였습니다."

        return answer

    def _generate_safe_fallback_answer(self, intent_type: str) -> str:
        """안전한 폴백 답변 생성 - 모든 의도에 대해 통일된 실패 메시지"""
        return "생성에 실패하였습니다."

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

    def _create_enhanced_korean_prompt(
        self,
        question: str,
        question_type: str,
        intent_analysis: Dict = None,
        domain_hints: Dict = None,
    ) -> str:
        """강화된 한국어 프롬프트 생성 - 템플릿 예시 활용"""
        domain = self._detect_domain(question)

        # 기본 한국어 전용 지시 (반복 방지 강화)
        korean_instruction = """
반드시 다음 규칙을 엄격히 준수하여 답변하세요:
1. 오직 완전한 한국어로만 답변 작성
2. 영어, 일본어, 중국어 등 외국어 절대 사용 금지
3. 깨진 문자나 특수 기호 사용 금지
4. 완전한 한국어 문장으로 구성
5. 문법에 맞는 자연스러운 한국어 표현 사용
6. 공백이 섞인 단어나 분리된 문자 사용 금지
7. 동일한 단어나 문구의 반복 절대 금지
8. 의미 없는 반복 문장 생성 금지
9. 논리적이고 일관된 내용으로 구성
10. 완전하고 명확한 문장으로 마무리
"""

        # 의도별 특화 지시 및 템플릿 예시
        intent_instruction = ""
        template_examples = ""

        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            if primary_intent in self.intent_specific_prompts:
                intent_instruction = random.choice(
                    self.intent_specific_prompts[primary_intent]
                )

            # 템플릿 예시 추가 (domain_hints에서 전달됨)
            if domain_hints and "template_examples" in domain_hints:
                examples = domain_hints["template_examples"]
                if examples and isinstance(examples, list):
                    # 2-3개의 예시를 선택하여 제공
                    selected_examples = examples[: min(3, len(examples))]

                    template_examples = "\n\n참고할 답변 예시 (이와 유사한 수준과 구조로 작성하되, 내용은 질문에 맞게 변형하세요):\n"
                    for i, example in enumerate(selected_examples, 1):
                        template_examples += f"예시 {i}: {example}\n"

                    template_examples += "\n위 예시들을 참고하여 질문의 내용에 맞는 구체적이고 전문적인 답변을 새롭게 작성하세요."

            # 답변 유형별 추가 지침
            if answer_type == "기관명":
                intent_instruction += (
                    "\n구체적인 기관명과 소속을 정확한 한국어로 명시하세요."
                )
            elif answer_type == "특징설명":
                intent_instruction += "\n주요 특징을 체계적으로 한국어로 나열하세요."
            elif answer_type == "지표나열":
                intent_instruction += "\n탐지 지표를 구체적으로 한국어로 설명하세요."
            elif answer_type == "방안제시":
                intent_instruction += "\n실무적 대응방안을 한국어로 제시하세요."

        # 힌트 정보 추가
        hint_context = ""
        if domain_hints:
            if (
                "institution_hints" in domain_hints
                and domain_hints["institution_hints"]
            ):
                hint_context += f"\n기관 정보: {domain_hints['institution_hints']}"
            if "improvement_type" in domain_hints:
                improvement_type = domain_hints["improvement_type"]
                if improvement_type == "korean_ratio_low":
                    hint_context += "\n한국어 비율을 높여 완전한 한국어로만 작성하세요."
                elif improvement_type == "intent_mismatch":
                    hint_context += f"\n질문 의도에 정확히 부합하는 답변을 작성하세요."

        if question_type == "multiple_choice":
            return self._create_enhanced_mc_prompt(
                question, self._extract_choice_count(question), domain, domain_hints
            )
        else:
            # 주관식 프롬프트 (템플릿 예시 포함)
            prompts = [
                f"""다음은 {domain} 분야의 금융보안 전문 질문입니다.

질문: {question}

{korean_instruction}

{intent_instruction}
{hint_context}
{template_examples}

전문가 수준의 정확한 답변을 완전한 한국어로만 작성하세요:
- 모든 전문 용어를 한국어로 표기
- 관련 법령과 규정을 한국어로 설명
- 체계적이고 논리적인 한국어 문장 구성
- 완전한 문장으로 마무리
- 동일한 표현의 반복 절대 금지
- 참고 예시의 수준과 구조를 유지하되 질문에 맞는 새로운 내용으로 작성

답변:""",
                f"""금융보안 전문가로서 다음 {domain} 관련 질문에 완전한 한국어로만 답변하세요.

{question}

{korean_instruction}

{intent_instruction}
{hint_context}
{template_examples}

답변 작성 기준:
- 반드시 한국어로만 작성
- 영어나 기타 외국어 사용 절대 금지
- 깨진 문자나 특수 기호 사용 금지
- 전문적이고 정확한 내용
- 완전한 문장으로 구성
- 논리적 흐름 유지
- 반복 표현 사용 금지
- 참고 예시를 바탕으로 질문에 특화된 답변 작성

답변:""",
            ]

            return random.choice(prompts)

    def _create_enhanced_mc_prompt(
        self,
        question: str,
        max_choice: int,
        domain: str = "일반",
        domain_hints: Dict = None,
    ) -> str:
        """강화된 객관식 프롬프트 생성"""
        if max_choice <= 0:
            max_choice = 5

        context = self._analyze_mc_context(question, domain)
        choice_range = f"1번부터 {max_choice}번 중"

        # 힌트 정보 추가
        hint_context = ""
        if (
            domain_hints
            and "pattern_hints" in domain_hints
            and domain_hints["pattern_hints"]
        ):
            hint_context = f"\n참고 정보: {domain_hints['pattern_hints']}"

        # 컨텍스트에 따른 지시사항
        if context["is_negative"]:
            instruction = (
                f"다음 중 해당하지 않거나 적절하지 않은 것을 {choice_range} 선택하세요."
            )
        elif context["is_positive"]:
            instruction = (
                f"다음 중 가장 적절하거나 옳은 것을 {choice_range} 선택하세요."
            )
        else:
            instruction = f"정답을 {choice_range} 선택하세요."

        return f"""다음은 {domain} 분야의 금융보안 객관식 문제입니다.

{question}
{hint_context}

{instruction}

각 선택지를 신중히 검토하고 정답 번호만 답하세요.
반드시 1부터 {max_choice}까지의 숫자 중 하나만 답하세요.

정답:"""

    def generate_answer(
        self,
        question: str,
        question_type: str,
        max_choice: int = 5,
        intent_analysis: Dict = None,
        domain_hints: Dict = None,
    ) -> str:
        """답변 생성 - 반드시 LLM 사용"""

        # 템플릿 예시를 domain_hints에 추가
        enhanced_domain_hints = domain_hints.copy() if domain_hints else {}

        if question_type == "subjective" and intent_analysis:
            # knowledge_base에서 템플릿 예시 가져오기
            domain = self._detect_domain(question)
            primary_intent = intent_analysis.get("primary_intent", "일반")

            # 의도 매핑
            intent_key = "일반"
            if "기관" in primary_intent:
                intent_key = "기관_묻기"
            elif "특징" in primary_intent:
                intent_key = "특징_묻기"
            elif "지표" in primary_intent:
                intent_key = "지표_묻기"
            elif "방안" in primary_intent:
                intent_key = "방안_묻기"
            elif "절차" in primary_intent:
                intent_key = "절차_묻기"
            elif "조치" in primary_intent:
                intent_key = "조치_묻기"

            # knowledge_base에서 템플릿 예시 가져오기 (임시로 여기서 직접 처리)
            template_examples = self._get_template_examples_from_knowledge(
                domain, intent_key
            )
            if template_examples:
                enhanced_domain_hints["template_examples"] = template_examples

        # 프롬프트 생성
        prompt = self._create_enhanced_korean_prompt(
            question, question_type, intent_analysis, enhanced_domain_hints
        )

        try:
            # 토크나이징 (한국어 최적화)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500,
                add_special_tokens=True,
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            # 생성 설정 (반복 방지 강화)
            gen_config = self._get_generation_config(question_type)

            # 모델 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # 반복 방지 강화
                    no_repeat_ngram_size=3,  # 3-gram 반복 방지
                )

            # 디코딩 (한국어 최적화)
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            # 반복 패턴 조기 감지
            if self.detect_repetitive_patterns(response):
                # 반복 패턴이 감지되면 다른 설정으로 재시도
                return self._retry_generation_with_different_settings(
                    prompt, question_type, max_choice, intent_analysis
                )

            # 후처리
            if question_type == "multiple_choice":
                answer = self._process_enhanced_mc_answer(
                    response, question, max_choice
                )
                success = answer and answer.isdigit() and 1 <= int(answer) <= max_choice
                self._add_learning_record(
                    question,
                    answer,
                    question_type,
                    success,
                    max_choice,
                    1.0 if success else 0.5,
                    intent_analysis,
                )
                return answer
            else:
                answer = self._process_enhanced_subj_answer(
                    response, question, intent_analysis
                )
                korean_ratio = self._calculate_korean_ratio(answer)
                quality_score = self._calculate_answer_quality(
                    answer, question, intent_analysis
                )
                success = korean_ratio > 0.8 and quality_score > 0.6

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
                question_type, question, max_choice, intent_analysis
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

    def _get_template_examples_from_knowledge(
        self, domain: str, intent_key: str
    ) -> List[str]:
        """지식베이스에서 템플릿 예시 가져오기 (임시 구현)"""
        # 이 부분은 실제로는 knowledge_base 인스턴스를 주입받아야 하지만
        # 임시로 기본 템플릿 예시를 제공합니다

        templates_mapping = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다.",
                ],
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다.",
                ],
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영됩니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                ],
            },
        }

        if domain in templates_mapping and intent_key in templates_mapping[domain]:
            return templates_mapping[domain][intent_key]

        return []

    def _retry_generation_with_different_settings(
        self,
        prompt: str,
        question_type: str,
        max_choice: int,
        intent_analysis: Dict = None,
    ) -> str:
        """다른 설정으로 재시도"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1000,
                add_special_tokens=True,
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            # 더 보수적인 생성 설정
            retry_config = GenerationConfig(
                max_new_tokens=200 if question_type == "subjective" else 10,
                temperature=0.4,  # 더 낮은 temperature
                top_p=0.7,  # 더 낮은 top_p
                do_sample=True,
                repetition_penalty=1.5,  # 더 강한 반복 방지
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=retry_config)

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            # 여전히 반복 패턴이 있다면 폴백 답변 사용
            if self.detect_repetitive_patterns(response):
                return "생성에 실패하였습니다."

            return response

        except Exception:
            # 최종 폴백
            return "생성에 실패하였습니다."

    def _process_enhanced_subj_answer(
        self, response: str, question: str, intent_analysis: Dict = None
    ) -> str:
        """강화된 주관식 답변 처리"""
        if not response:
            return ""

        # 1단계: 반복 패턴 조기 감지 및 처리
        if self.detect_repetitive_patterns(response):
            response = self.remove_repetitive_patterns(response)
            # 너무 많이 제거되어 내용이 부족한 경우
            if len(response) < 30:
                return "생성에 실패하였습니다."

        # 2단계: 한국어 텍스트 복구
        response = self.recover_korean_text(response)

        # 3단계: 품질 향상
        response = self.enhance_korean_answer_quality(
            response, question, intent_analysis
        )

        # 4단계: 기본 정리
        response = re.sub(r"\s+", " ", response).strip()

        # 5단계: 불필요한 내용 제거
        # 프롬프트 관련 내용 제거
        response = re.sub(r"답변[:：]\s*", "", response)
        response = re.sub(r"질문[:：].*?\n", "", response)
        response = re.sub(r"다음.*?답변하세요[.:]\s*", "", response)

        # 6단계: 한국어 검증
        korean_ratio = self._calculate_korean_ratio(response)

        # 7단계: 의도별 답변 검증 및 개선
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            # 기관명이 필요한 경우
            if answer_type == "기관명":
                institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
                if not any(keyword in response for keyword in institution_keywords):
                    # 질문 내용을 바탕으로 적절한 기관명 추가
                    if "전자금융" in question and "분쟁" in question:
                        response = "전자금융분쟁조정위원회에서 " + response
                    elif "개인정보" in question:
                        response = "개인정보보호위원회에서 " + response
                    elif "한국은행" in question:
                        response = "한국은행에서 " + response

        # 8단계: 최종 검증 및 보완
        if korean_ratio < 0.7 or len(response) < 30:
            # 기본 응답으로 대체
            response = "생성에 실패하였습니다."

        # 9단계: 길이 조절
        if len(response) > 350:
            sentences = response.split(". ")
            response = ". ".join(sentences[:3])
            if not response.endswith("."):
                response += "."

        # 10단계: 마침표 확인
        if (
            response
            and not response.endswith((".", "다", "요", "함"))
            and response != "생성에 실패하였습니다."
        ):
            response += "."

        # 11단계: 최종 반복 패턴 확인
        if self.detect_repetitive_patterns(response):
            # 최종적으로도 반복 패턴이 있다면 실패 메시지로 대체
            return "생성에 실패하였습니다."

        return response

    def _process_enhanced_mc_answer(
        self, response: str, question: str, max_choice: int
    ) -> str:
        """강화된 객관식 답변 처리"""
        if max_choice <= 0:
            max_choice = 5

        # 1단계: 텍스트 복구
        response = self.recover_korean_text(response)

        # 2단계: 숫자 추출 (선택지 범위 내에서만)
        numbers = re.findall(r"[1-9]", response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.answer_distributions:
                    self.answer_distributions[max_choice][num] += 1
                    self.mc_answer_counts[max_choice] += 1
                return num

        # 3단계: 유효한 답변을 찾지 못한 경우 강제 생성
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

    def generate_contextual_mc_answer(
        self, question: str, max_choice: int, domain: str
    ) -> str:
        """컨텍스트 기반 객관식 답변 생성"""
        context_hints = self._analyze_mc_context(question, domain)
        prompt = self._create_enhanced_mc_prompt(
            question, max_choice, domain, {"context_hints": context_hints}
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1000
            )
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            gen_config = self._get_generation_config("multiple_choice")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            answer = self._process_enhanced_mc_answer(response, question, max_choice)

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
        """개선된 답변 생성"""
        return self.generate_answer(
            question, question_type, max_choice, intent_analysis, improvement_hints
        )

    def generate_fallback_mc_answer(
        self, question: str, max_choice: int, domain: str
    ) -> str:
        """폴백 객관식 답변 생성"""
        return self.generate_contextual_mc_answer(question, max_choice, domain)

    def generate_fallback_subjective_answer(self, question: str) -> str:
        """폴백 주관식 답변 생성"""
        domain = self._detect_domain(question)
        prompt = self._create_enhanced_korean_prompt(
            question, "subjective", None, {"fallback_mode": True}
        )

        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1000
            )
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            gen_config = self._get_generation_config("subjective")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            processed_response = self._process_enhanced_subj_answer(
                response, question, None
            )

            # 반복 패턴이 있다면 실패 메시지로 대체
            if self.detect_repetitive_patterns(processed_response):
                return "생성에 실패하였습니다."

            return processed_response

        except Exception as e:
            if self.verbose:
                print(f"폴백 주관식 답변 생성 오류: {e}")
            return "생성에 실패하였습니다."

    def _generate_basic_intent_answer(self, primary_intent: str) -> str:
        """기본 의도별 답변 생성"""
        return "생성에 실패하였습니다."

    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정 (반복 방지 강화)"""
        config_dict = GENERATION_CONFIG[question_type].copy()
        config_dict["pad_token_id"] = self.tokenizer.pad_token_id
        config_dict["eos_token_id"] = self.tokenizer.eos_token_id

        # 반복 방지 설정 추가
        if question_type == "subjective":
            config_dict["repetition_penalty"] = 1.2
            config_dict["no_repeat_ngram_size"] = 3
        else:
            config_dict["repetition_penalty"] = 1.1
            config_dict["no_repeat_ngram_size"] = 2

        return GenerationConfig(**config_dict)

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

    def _calculate_answer_quality(
        self, answer: str, question: str, intent_analysis: Dict = None
    ) -> float:
        """답변 품질 점수 계산"""
        if not answer:
            return 0.0

        score = 0.0

        # 반복 패턴 페널티
        if self.detect_repetitive_patterns(answer):
            return 0.1  # 반복 패턴이 있으면 매우 낮은 점수

        # 한국어 비율 (30%)
        korean_ratio = self._calculate_korean_ratio(answer)
        score += korean_ratio * 0.3

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

        # 의도 일치성 (25%)
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            if self._check_intent_match(answer, answer_type):
                score += 0.25
            else:
                score += 0.1
        else:
            score += 0.15

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
    ) -> str:
        """폴백 답변 생성"""
        if question_type == "multiple_choice":
            if max_choice <= 0:
                max_choice = 5
            domain = self._detect_domain(question)
            return self.generate_fallback_mc_answer(question, max_choice, domain)
        else:
            return self.generate_fallback_subjective_answer(question)

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
        # 반복 패턴이 있는 답변은 실패로 기록
        if self.detect_repetitive_patterns(answer):
            success = False
            quality_score = min(quality_score, 0.2)

        record = {
            "question": question[:200],
            "answer": answer[:300],
            "type": question_type,
            "max_choice": max_choice,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "has_repetition": self.detect_repetitive_patterns(answer),
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
                    "has_repetition": self.detect_repetitive_patterns(answer),
                }
                self.learning_data["intent_based_answers"][primary_intent].append(
                    intent_record
                )

                # 최근 50개만 유지
                if len(self.learning_data["intent_based_answers"][primary_intent]) > 50:
                    self.learning_data["intent_based_answers"][primary_intent] = (
                        self.learning_data["intent_based_answers"][primary_intent][-50:]
                    )

                # 고품질 답변은 템플릿으로 저장 (반복 없는 것만)
                if quality_score > 0.85 and not self.detect_repetitive_patterns(answer):
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
                        "has_repetition": False,
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
                "repetition_count": 0,
            }

        patterns = self.learning_data["question_patterns"][domain]
        patterns["count"] += 1
        patterns["avg_quality"] = (
            patterns["avg_quality"] * (patterns["count"] - 1) + quality_score
        ) / patterns["count"]

        if self.detect_repetitive_patterns(answer):
            patterns["repetition_count"] += 1

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
                    do_sample=False,
                    repetition_penalty=1.1,
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
        }

    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        # 반복 패턴 통계 추가
        repetition_stats = {
            "total_repetitive_answers": sum(
                1
                for record in self.learning_data["successful_answers"]
                + self.learning_data["failed_answers"]
                if record.get("has_repetition", False)
            ),
            "repetition_rate_by_domain": {
                domain: (
                    patterns.get("repetition_count", 0) / max(patterns["count"], 1)
                )
                * 100
                for domain, patterns in self.learning_data["question_patterns"].items()
            },
        }

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
            "repetition_stats": repetition_stats,
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
