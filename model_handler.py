# model_handler.py

import torch
import re
import time
import gc
import random
import os
import json
import unicodedata
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from config import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG,
    GENERATION_CONFIG,
    OPTIMIZATION_CONFIG,
    JSON_CONFIG_FILES,
    MEMORY_CONFIG,
    get_device,
)


class SimpleModelHandler:

    def __init__(self, model_name: str = None, verbose: bool = False):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.verbose = verbose
        self.device = get_device()

        self._load_json_configs()

        self.optimization_config = OPTIMIZATION_CONFIG

        if verbose:
            print(f"모델 로딩: {self.model_name}")
            print(f"디바이스: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
            use_fast=MODEL_CONFIG["use_fast_tokenizer"],
        )

        self._optimize_tokenizer_for_korean()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
            device_map=MODEL_CONFIG["device_map"],
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
        )

        self.model.eval()

        self._warmup()

        if verbose:
            print("모델 로딩 완료")

    def _optimize_tokenizer_for_korean(self):
        if hasattr(self.tokenizer, "do_lower_case"):
            self.tokenizer.do_lower_case = False

        if hasattr(self.tokenizer, "normalize"):
            self.tokenizer.normalize = False

        special_tokens = ["<korean>", "</korean>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _load_json_configs(self):
        try:
            with open(JSON_CONFIG_FILES["model_config"], "r", encoding="utf-8") as f:
                model_config = json.load(f)

            with open(
                JSON_CONFIG_FILES["processing_config"], "r", encoding="utf-8"
            ) as f:
                processing_config = json.load(f)

            self.mc_context_patterns = model_config["mc_context_patterns"]
            self.intent_specific_prompts = model_config["intent_specific_prompts"]

            self.korean_recovery_config = processing_config["korean_text_recovery"]
            self.korean_quality_patterns = processing_config["korean_quality_patterns"]

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
        self.korean_recovery_mapping = {}

        for broken, replacement in self.korean_recovery_config[
            "broken_unicode_chars"
        ].items():
            try:
                actual_char = broken.encode().decode("unicode_escape")
                self.korean_recovery_mapping[actual_char] = replacement
            except:
                pass

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["japanese_katakana_removal"]
        )

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["broken_korean_patterns"]
        )

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["spaced_korean_fixes"]
        )

        self.korean_recovery_mapping.update(
            self.korean_recovery_config["common_korean_typos"]
        )

    def _load_default_configs(self):
        print("기본 설정으로 대체합니다.")

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

        self.korean_recovery_mapping = {
            "어어지인": "",
            "선 어": "",
            "언 어": "",
            "순 어": "",
            "ᄒᆞᆫ": "",
            "작로": "으로",
            "갈취 묻는 말": "",
            "묻고 갈취": "",
            "갈취": "",
        }

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

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        if not text or len(text) < 20:
            return False

        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취",
            r"(.{1,3})\s*(\1\s*){10,}",
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

        words = text.split()
        if len(words) >= 10:
            for i in range(len(words) - 9):
                same_count = 1
                for j in range(i + 1, min(i + 10, len(words))):
                    if words[i] == words[j]:
                        same_count += 1
                    else:
                        break

                if same_count >= 10:
                    return True

        return False

    def remove_repetitive_patterns(self, text: str) -> str:
        if not text:
            return ""

        problematic_removals = [
            "갈취 묻는 말",
            "묻고 갈취",
        ]

        for pattern in problematic_removals:
            text = text.replace(pattern, "")

        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            count = 1
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            if count >= 8:
                cleaned_words.extend([current_word] * min(5, count))
            else:
                cleaned_words.extend([current_word] * count)

            i += count

        text = " ".join(cleaned_words)

        text = re.sub(r"(.{5,15})\s*\1\s*\1\s*\1\s*\1+", r"\1", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def recover_korean_text(self, text: str) -> str:
        if not text:
            return ""

        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_repetitive_patterns(text)

        text = unicodedata.normalize("NFC", text)

        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        for pattern_config in self.korean_quality_patterns:
            pattern = pattern_config["pattern"]
            replacement = pattern_config["replacement"]
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def enhance_korean_answer_quality(
        self, answer: str, question: str = "", intent_analysis: Dict = None
    ) -> str:
        if not answer:
            return ""

        if self.detect_critical_repetitive_patterns(answer):
            answer = self.remove_repetitive_patterns(answer)
            if len(answer) < 15:
                return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        answer = self.recover_korean_text(answer)

        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            if answer_type == "기관명":
                institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
                if not any(keyword in answer for keyword in institution_keywords):
                    if "전자금융" in question or "분쟁조정" in question:
                        answer = "전자금융분쟁조정위원회에서 " + answer
                    elif "개인정보" in question:
                        answer = "개인정보보호위원회에서 " + answer

            elif answer_type == "특징설명":
                if "특징" not in answer and "특성" not in answer:
                    answer = "주요 특징은 " + answer

            elif answer_type == "지표나열":
                if "지표" not in answer and "탐지" not in answer:
                    answer = "주요 탐지 지표는 " + answer

        if len(answer) > 10 and not answer.endswith((".", "다", "요", "함")):
            if answer.endswith("니"):
                answer += "다."
            elif answer.endswith("습"):
                answer += "니다."
            else:
                answer += "."

        if len(answer) > 600:
            sentences = answer.split(". ")
            if len(sentences) > 5:
                answer = ". ".join(sentences[:5])
                if not answer.endswith("."):
                    answer += "."

        answer = re.sub(r"\s+", " ", answer).strip()

        if self.detect_critical_repetitive_patterns(answer):
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립해야 합니다."

        return answer

    def _generate_safe_fallback_answer(self, intent_type: str) -> str:
        fallback_templates = {
            "기관_묻기": "관련 전문 기관에서 해당 업무를 담당하고 있습니다.",
            "특징_묻기": "주요 특징을 체계적으로 분석하여 관리해야 합니다.",
            "지표_묻기": "주요 탐지 지표를 통해 모니터링과 분석을 수행해야 합니다.",
            "방안_묻기": "체계적인 대응 방안을 수립하고 실행해야 합니다.",
            "절차_묻기": "관련 절차에 따라 단계별로 수행해야 합니다.",
            "조치_묻기": "적절한 보안 조치를 시행해야 합니다.",
        }

        return fallback_templates.get(
            intent_type, "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."
        )

    def _extract_choice_count(self, question: str) -> int:
        lines = question.split("\n")
        choice_numbers = []

        for line in lines:
            match = re.match(r"^(\d+)\s+(.+)", line.strip())
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_numbers.append(num)

        if choice_numbers:
            choice_numbers.sort()
            return max(choice_numbers)

        for i in range(5, 2, -1):
            pattern = r"1\s.*" + ".*".join([f"{j}\s" for j in range(2, i + 1)])
            if re.search(pattern, question, re.DOTALL):
                return i

        return 5

    def _analyze_mc_context(self, question: str, domain: str = "일반") -> Dict:
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

        for pattern in self.mc_context_patterns["negative_keywords"]:
            if re.search(pattern, question_lower):
                context["is_negative"] = True
                break

        for pattern in self.mc_context_patterns["positive_keywords"]:
            if re.search(pattern, question_lower):
                context["is_positive"] = True
                break

        if domain in self.mc_context_patterns["domain_specific_patterns"]:
            domain_info = self.mc_context_patterns["domain_specific_patterns"][domain]

            keyword_matches = sum(
                1 for keyword in domain_info["keywords"] if keyword in question_lower
            )

            if keyword_matches > 0:
                context["domain_hints"].append(domain)
                context["likely_answers"] = domain_info["common_answers"]
                context["confidence_score"] = min(
                    keyword_matches / len(domain_info["keywords"]), 1.0
                )

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
        domain = self._detect_domain(question)

        template_examples_text = ""
        if domain_hints and "template_examples" in domain_hints:
            examples = domain_hints["template_examples"]
            if examples and isinstance(examples, list) and len(examples) > 0:
                selected_examples = examples[:3]
                template_examples_text = (
                    "\n\n=== 참고 예시 (이와 유사한 수준과 구조로 작성하세요) ===\n"
                )
                for i, example in enumerate(selected_examples, 1):
                    template_examples_text += f"\n예시 {i}: {example}\n"
                template_examples_text += "\n위 예시들을 참고하여 질문에 적합한 구체적이고 전문적인 답변을 작성하세요.\n"
                template_examples_text += "예시와 비슷한 길이, 구조, 전문성 수준으로 답변하되 질문 내용에 맞게 작성하세요.\n"

        korean_instruction = """
다음 규칙을 준수하여 답변하세요:
1. 완전한 한국어로만 답변 작성
2. 전문적이고 구체적인 내용으로 구성
3. 자연스러운 한국어 표현 사용
4. 논리적이고 일관된 설명
5. 완전한 문장으로 마무리
"""

        intent_instruction = ""

        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            if primary_intent in self.intent_specific_prompts:
                intent_instruction = random.choice(
                    self.intent_specific_prompts[primary_intent]
                )

            if answer_type == "기관명":
                intent_instruction += (
                    "\n구체적인 기관명과 소속을 정확한 한국어로 명시하세요."
                )
            elif answer_type == "특징설명":
                intent_instruction += (
                    "\n주요 특징을 체계적으로 한국어로 나열하고 상세히 설명하세요."
                )
            elif answer_type == "지표나열":
                intent_instruction += "\n탐지 지표를 구체적으로 한국어로 설명하고 실무적 관점에서 제시하세요."
            elif answer_type == "방안제시":
                intent_instruction += (
                    "\n실무적 대응방안을 단계별로 한국어로 제시하세요."
                )

        hint_context = ""
        if domain_hints:
            if (
                "institution_hints" in domain_hints
                and domain_hints["institution_hints"]
            ):
                hint_context += f"\n기관 정보: {domain_hints['institution_hints']}"

        if question_type == "multiple_choice":
            return self._create_enhanced_mc_prompt(
                question, self._extract_choice_count(question), domain, domain_hints
            )
        else:
            prompt_template = f"""다음은 {domain} 분야의 금융보안 전문 질문입니다.

질문: {question}

{korean_instruction}
{intent_instruction}
{hint_context}
{template_examples_text}

위의 참고 예시들과 비슷한 수준의 전문성과 구체성으로 답변을 작성하세요.
반드시 완전한 한국어로만 작성하고, 법령과 규정을 근거로 한 실무적 내용을 포함하세요.

답변:"""

            return prompt_template

    def _create_enhanced_mc_prompt(
        self,
        question: str,
        max_choice: int,
        domain: str = "일반",
        domain_hints: Dict = None,
    ) -> str:
        if max_choice <= 0:
            max_choice = 5

        context = self._analyze_mc_context(question, domain)
        choice_range = f"1번부터 {max_choice}번 중"

        hint_context = ""
        if (
            domain_hints
            and "pattern_hints" in domain_hints
            and domain_hints["pattern_hints"]
        ):
            hint_context = f"\n참고 정보: {domain_hints['pattern_hints']}"

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

        enhanced_domain_hints = domain_hints.copy() if domain_hints else {}

        if question_type == "subjective" and intent_analysis:
            domain = self._detect_domain(question)
            primary_intent = intent_analysis.get("primary_intent", "일반")

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

            template_examples = self._get_template_examples_from_knowledge(
                domain, intent_key
            )
            if template_examples:
                enhanced_domain_hints["template_examples"] = template_examples

        prompt = self._create_enhanced_korean_prompt(
            question, question_type, intent_analysis, enhanced_domain_hints
        )

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2000,
                add_special_tokens=True,
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            gen_config = self._get_generation_config(question_type)

            if question_type == "subjective":
                gen_config.max_new_tokens = 400
                gen_config.repetition_penalty = 1.05
                gen_config.no_repeat_ngram_size = 2
                gen_config.temperature = 0.7
                gen_config.top_p = 0.95
                gen_config.length_penalty = 1.0

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if self.detect_critical_repetitive_patterns(response):
                return self._retry_generation_with_different_settings(
                    prompt, question_type, max_choice, intent_analysis
                )

            if question_type == "multiple_choice":
                answer = self._process_enhanced_mc_answer(
                    response, question, max_choice
                )
                return answer
            else:
                answer = self._process_enhanced_subj_answer(
                    response, question, intent_analysis
                )
                return answer

        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")

            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                fallback = self._generate_safe_fallback_answer(primary_intent)
            else:
                fallback = self._get_fallback_answer_with_llm(
                    question_type, question, max_choice, intent_analysis
                )

            return fallback

    def _get_template_examples_from_knowledge(
        self, domain: str, intent_key: str
    ) -> List[str]:
        templates_mapping = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "트로이 목마는 유용한 프로그램으로 가장하여 사용자가 직접 설치하도록 유도하고, 설치 후 악의적인 기능을 수행하는 특징을 가집니다. 원격 접근 기능을 통해 시스템을 외부에서 조작할 수 있습니다.",
                    "원격접근 도구의 주요 특징은 은밀한 설치, 지속적인 연결 유지, 시스템 전반에 대한 제어권 획득, 사용자 모르게 정보 수집 등이며, 탐지를 회피하기 위한 다양한 기법을 사용합니다.",
                    "악성 원격접근 도구는 정상 소프트웨어로 위장하여 배포되며, 설치 후 시스템 권한을 탈취하고 외부 서버와 은밀한 통신을 수행하는 특성을 가집니다.",
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다.",
                    "비정상적인 아웃바운드 연결, 시스템 리소스 과다 사용, 백그라운드 프로세스 증가, 보안 소프트웨어 비활성화 시도, 시스템 설정 변경 등의 징후를 종합적으로 분석해야 합니다.",
                    "네트워크 연결 로그 분석, 프로세스 모니터링, 파일 무결성 검사, 레지스트리 변경 감시, 시스템 콜 추적 등을 통해 악성 활동을 탐지할 수 있습니다.",
                ],
                "방안_묻기": [
                    "딥페이크 기술 악용에 대비하여 다층 방어체계 구축, 실시간 딥페이크 탐지 시스템 도입, 직원 교육 및 인식 개선, 생체인증 강화, 다중 인증 체계 구축 등의 종합적 대응방안이 필요합니다.",
                    "네트워크 분할을 통한 격리, 접근권한 최소화 원칙 적용, 행위 기반 탐지 시스템 구축, 사고 대응 절차 수립, 백업 및 복구 체계 마련 등의 보안 강화 방안을 수립해야 합니다.",
                    "엔드포인트 보안 강화, 네트워크 트래픽 모니터링, 사용자 인식 개선 교육, 보안 정책 수립 및 준수, 정기적인 보안 점검 등을 통해 종합적인 보안 관리체계를 구축해야 합니다.",
                    "SBOM 활용을 통한 소프트웨어 공급망 보안 강화, 구성 요소 취약점 관리, 라이선스 컴플라이언스 확보, 보안 업데이트 추적 관리 등의 방안을 수립해야 합니다.",
                    "인공지능 기반 이상 행위 탐지, 실시간 모니터링 체계 구축, 사용자 행위 분석, 보안 인식 교육 강화, 다중 인증 시스템 도입 등의 대응방안을 마련해야 합니다.",
                ],
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다.",
                    "개인정보 침해 관련 신고 및 상담은 개인정보보호위원회 산하 개인정보침해신고센터에서 담당하고 있습니다.",
                    "개인정보 관련 분쟁의 조정은 개인정보보호위원회 내 개인정보 분쟁조정위원회에서 담당하며, 피해구제와 분쟁해결 업무를 수행합니다.",
                    "개인정보보호 정책 수립과 법령 집행은 개인정보보호위원회에서 담당하고, 침해신고 접수와 상담은 개인정보침해신고센터에서 처리합니다.",
                ],
                "방안_묻기": [
                    "개인정보 처리 시 수집 최소화 원칙 적용, 목적 외 이용 금지, 적절한 보호조치 수립, 정기적인 개인정보 영향평가 실시, 정보주체 권리 보장 체계 구축 등의 관리방안이 필요합니다.",
                    "개인정보보호 관리체계 구축, 개인정보처리방침 수립 및 공개, 개인정보보호책임자 지정, 정기적인 교육 실시, 기술적·관리적·물리적 보호조치 이행 등을 체계적으로 수행해야 합니다.",
                    "개인정보 수집 시 동의 절차 준수, 처리목적 명확화, 보유기간 설정 및 준수, 정보주체 권리 행사 절차 마련, 개인정보 파기 체계 구축 등의 전 과정 관리방안을 수립해야 합니다.",
                    "만 14세 미만 아동의 개인정보 처리 시 법정대리인의 동의 확보, 아동의 인지 능력을 고려한 처리 방안 수립, 특별한 보호조치 마련 등이 필요합니다.",
                ],
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영됩니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다.",
                    "전자금융거래법에 따라 금융감독원의 전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁의 조정 업무를 담당하고 있습니다.",
                    "전자금융 분쟁조정은 금융감독원에 설치된 전자금융분쟁조정위원회에서 신청할 수 있으며, 이용자 보호를 위한 분쟁해결 업무를 수행합니다.",
                    "전자금융거래 분쟁의 조정은 금융감독원 전자금융분쟁조정위원회에서 담당하며, 공정하고 신속한 분쟁해결을 위한 업무를 수행합니다.",
                ],
                "방안_묻기": [
                    "접근매체 보안 강화, 전자서명 및 인증체계 고도화, 거래내역 실시간 통지, 이상거래 탐지시스템 구축, 이용자 교육 강화 등의 종합적인 보안방안이 필요합니다.",
                    "전자금융업자의 보안조치 의무 강화, 이용자 피해보상 체계 개선, 분쟁조정 절차 신속화, 보안기술 표준화, 관련 법령 정비 등의 제도적 개선방안을 추진해야 합니다.",
                    "다중 인증 체계 도입, 거래한도 설정 및 관리, 보안카드 및 이용자 신원확인 강화, 금융사기 예방 시스템 구축, 이용자 보호 교육 확대 등을 실시해야 합니다.",
                ],
            },
            "정보보안": {
                "방안_묻기": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다.",
                    "정보자산 분류체계 구축, 중요도에 따른 차등 보안조치 적용, 정기적인 보안교육과 인식제고 프로그램 운영, 보안사고 대응체계 구축 등이 필요합니다.",
                    "물리적 보안조치, 기술적 보안조치, 관리적 보안조치를 균형있게 적용하고, 지속적인 보안성 평가와 개선활동을 수행해야 합니다.",
                ],
            },
            "금융투자": {
                "방안_묻기": [
                    "투자자 보호를 위한 적합성 원칙 준수, 투자위험 고지 의무 이행, 투자자문 서비스 품질 개선, 불공정거래 방지 체계 구축, 내부통제 시스템 강화 등의 방안이 필요합니다.",
                    "금융투자업자의 영업행위 규준 강화, 투자자 교육 확대, 분쟁조정 절차 개선, 시장감시 체계 고도화, 투자자 보호기금 운영 내실화 등을 추진해야 합니다.",
                    "투자상품 설명의무 강화, 투자자 유형별 맞춤형 서비스 제공, 투자권유 과정의 투명성 제고, 이해상충 방지 체계 구축, 투자자 피해구제 절차 개선 등이 필요합니다.",
                ],
            },
            "위험관리": {
                "방안_묻기": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다.",
                    "위험관리 정책과 절차를 수립하고 위험한도를 설정하여 관리하며, 위험관리 조직과 책임체계를 명확히 정의해야 합니다.",
                    "위험관리 문화 조성, 위험관리 교육 강화, 위험보고 체계 구축, 위험관리 성과평가 체계 도입, 외부 위험요인 모니터링 강화 등을 실시해야 합니다.",
                ],
            },
        }

        general_templates = {
            "특징_묻기": [
                "주요 특징을 체계적으로 분석하여 관리해야 합니다.",
                "핵심적인 특성과 성질을 정확히 파악하여 대응해야 합니다.",
                "해당 분야의 주요 특징은 체계적인 접근과 지속적인 관리를 통해 효과적으로 처리할 수 있습니다.",
            ],
            "지표_묻기": [
                "주요 탐지 지표를 통해 체계적인 모니터링과 분석을 수행해야 합니다.",
                "관련 징후와 패턴을 분석하여 적절한 대응조치를 시행해야 합니다.",
                "실시간 모니터링과 정기적인 점검을 통해 이상 징후를 조기에 발견하고 대응해야 합니다.",
            ],
            "방안_묻기": [
                "체계적인 대응 방안을 수립하고 관련 법령에 따라 지속적으로 관리해야 합니다.",
                "효과적인 관리 방안을 마련하여 정기적인 점검과 개선을 수행해야 합니다.",
                "종합적인 대응체계를 구축하고 단계별 실행계획을 수립하여 체계적으로 관리해야 합니다.",
            ],
            "기관_묻기": [
                "관련 전문 기관에서 해당 업무를 법령에 따라 담당하고 있습니다.",
                "소관 기관에서 체계적인 관리와 감독 업무를 수행하고 있습니다.",
            ],
            "절차_묻기": [
                "관련 절차에 따라 단계별로 체계적인 수행과 관리가 필요합니다.",
                "법령에 정해진 절차를 준수하여 순차적으로 진행해야 합니다.",
            ],
            "조치_묻기": [
                "적절한 보안 조치를 시행하고 관련 법령에 따라 지속적으로 관리해야 합니다.",
                "필요한 조치사항을 파악하여 체계적인 대응과 개선을 수행해야 합니다.",
            ],
        }

        if domain in templates_mapping and intent_key in templates_mapping[domain]:
            return templates_mapping[domain][intent_key]

        return general_templates.get(
            intent_key,
            [
                "관련 법령과 규정에 따라 체계적인 관리가 필요합니다.",
                "해당 분야의 전문적 지식을 바탕으로 적절한 대응을 수행해야 합니다.",
            ],
        )

    def _retry_generation_with_different_settings(
        self,
        prompt: str,
        question_type: str,
        max_choice: int,
        intent_analysis: Dict = None,
    ) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500,
                add_special_tokens=True,
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            retry_config = GenerationConfig(
                max_new_tokens=350 if question_type == "subjective" else 10,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=retry_config)

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            if self.detect_critical_repetitive_patterns(response):
                if intent_analysis:
                    primary_intent = intent_analysis.get("primary_intent", "일반")
                    return self._generate_safe_fallback_answer(primary_intent)
                return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

            return response

        except Exception:
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

    def _process_enhanced_subj_answer(
        self, response: str, question: str, intent_analysis: Dict = None
    ) -> str:
        if not response:
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        if self.detect_critical_repetitive_patterns(response):
            response = self.remove_repetitive_patterns(response)
            if len(response) < 15:
                if intent_analysis:
                    primary_intent = intent_analysis.get("primary_intent", "일반")
                    return self._generate_safe_fallback_answer(primary_intent)
                return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        response = self.recover_korean_text(response)

        response = self.enhance_korean_answer_quality(
            response, question, intent_analysis
        )

        response = re.sub(r"\s+", " ", response).strip()

        response = re.sub(r"답변[:：]\s*", "", response)
        response = re.sub(r"질문[:：].*?\n", "", response)
        response = re.sub(r"다음.*?답변하세요[.:]\s*", "", response)

        korean_ratio = self._calculate_korean_ratio(response)

        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            if answer_type == "기관명":
                institution_keywords = ["위원회", "감독원", "은행", "기관", "센터"]
                if not any(keyword in response for keyword in institution_keywords):
                    if "전자금융" in question and "분쟁" in question:
                        response = "전자금융분쟁조정위원회에서 " + response
                    elif "개인정보" in question:
                        response = "개인정보보호위원회에서 " + response
                    elif "한국은행" in question:
                        response = "한국은행에서 " + response

        if korean_ratio < 0.4 or len(response) < 10:
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                response = self._generate_safe_fallback_answer(primary_intent)
            else:
                response = "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        if len(response) > 500:
            sentences = response.split(". ")
            response = ". ".join(sentences[:5])
            if not response.endswith("."):
                response += "."

        if (
            response
            and not response.endswith((".", "다", "요", "함"))
            and "생성에 실패" not in response
            and "관리가 필요" not in response
        ):
            response += "."

        if self.detect_critical_repetitive_patterns(response):
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        return response

    def _process_enhanced_mc_answer(
        self, response: str, question: str, max_choice: int
    ) -> str:
        if max_choice <= 0:
            max_choice = 5

        response = self.recover_korean_text(response)

        numbers = re.findall(r"[1-9]", response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                return num

        return self._force_valid_mc_answer(response, max_choice)

    def _force_valid_mc_answer(self, response: str, max_choice: int) -> str:
        if max_choice <= 0:
            max_choice = 5

        all_numbers = re.findall(r"\d+", response)

        for num_str in all_numbers:
            num = int(num_str)
            if 1 <= num <= max_choice:
                return str(num)

        return str((max_choice + 1) // 2)

    def generate_contextual_mc_answer(
        self, question: str, max_choice: int, domain: str
    ) -> str:
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

    def generate_fallback_mc_answer(
        self, question: str, max_choice: int, domain: str
    ) -> str:
        return self.generate_contextual_mc_answer(question, max_choice, domain)

    def generate_fallback_subjective_answer(self, question: str) -> str:
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
            gen_config.repetition_penalty = 1.05
            gen_config.no_repeat_ngram_size = 2
            gen_config.temperature = 0.8

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            processed_response = self._process_enhanced_subj_answer(
                response, question, None
            )

            if self.detect_critical_repetitive_patterns(processed_response):
                return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

            return processed_response

        except Exception as e:
            if self.verbose:
                print(f"폴백 주관식 답변 생성 오류: {e}")
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        config_dict = GENERATION_CONFIG[question_type].copy()
        config_dict["pad_token_id"] = self.tokenizer.pad_token_id
        config_dict["eos_token_id"] = self.tokenizer.eos_token_id

        if question_type == "subjective":
            config_dict["repetition_penalty"] = 1.05
            config_dict["no_repeat_ngram_size"] = 2
            config_dict["temperature"] = 0.7
            config_dict["top_p"] = 0.95
            config_dict["max_new_tokens"] = 400
        else:
            config_dict["repetition_penalty"] = 1.1
            config_dict["no_repeat_ngram_size"] = 2

        return GenerationConfig(**config_dict)

    def _detect_domain(self, question: str) -> str:
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
        if not text:
            return 0.0

        korean_chars = len(re.findall(r"[가-힣]", text))
        total_chars = len(re.sub(r"[^\w가-힣]", "", text))

        if total_chars == 0:
            return 0.0

        return korean_chars / total_chars

    def _get_domain_keywords(self, question: str) -> List[str]:
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
        if question_type == "multiple_choice":
            if max_choice <= 0:
                max_choice = 5
            domain = self._detect_domain(question)
            return self.generate_fallback_mc_answer(question, max_choice, domain)
        else:
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return self.generate_fallback_subjective_answer(question)

    def _warmup(self):
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

    def cleanup(self):
        try:
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
