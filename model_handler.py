# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 질문 의도 기반 답변 생성
"""

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

# 설정 파일 import
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
    """모델 핸들러"""

    def __init__(self, model_name: str = None, verbose: bool = False):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.verbose = verbose
        self.device = get_device()

        # JSON 설정 파일에서 데이터 로드
        self._load_json_configs()

        # 성능 최적화 설정
        self.optimization_config = OPTIMIZATION_CONFIG

        if verbose:
            print(f"모델 로딩: {self.model_name}")
            print(f"디바이스: {self.device}")

        # 토크나이저 로드
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

    def _optimize_tokenizer_for_korean(self):
        """토크나이저 한국어 최적화"""
        if hasattr(self.tokenizer, "do_lower_case"):
            self.tokenizer.do_lower_case = False

        if hasattr(self.tokenizer, "normalize"):
            self.tokenizer.normalize = False

        # 특수 토큰 추가
        special_tokens = ["<korean>", "</korean>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _load_json_configs(self):
        """JSON 설정 파일 로드"""
        try:
            # model_config.json 로드
            with open(JSON_CONFIG_FILES["model_config"], "r", encoding="utf-8") as f:
                model_config = json.load(f)

            # processing_config.json 로드
            with open(
                JSON_CONFIG_FILES["processing_config"], "r", encoding="utf-8"
            ) as f:
                processing_config = json.load(f)

            # 모델 관련 데이터 할당
            self.mc_context_patterns = model_config["mc_context_patterns"]
            self.intent_specific_prompts = model_config["intent_specific_prompts"]

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
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {}

        # 깨진 유니코드 문자 제거
        for broken, replacement in self.korean_recovery_config[
            "broken_unicode_chars"
        ].items():
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
        """기본 설정 로드"""
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

        # 기본 한국어 복구 매핑
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

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """치명적인 반복 패턴 감지"""
        if not text or len(text) < 20:
            return False

        # 치명적인 반복 패턴만 감지
        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취", 
            r"(.{1,3})\s*(\1\s*){8,}",  # 8회 이상 반복만 감지
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

        # 같은 단어가 연속으로 8번 이상 나오는 경우만 감지
        words = text.split()
        if len(words) >= 8:
            for i in range(len(words) - 7):
                same_count = 1
                for j in range(i + 1, min(i + 8, len(words))):
                    if words[i] == words[j]:
                        same_count += 1
                    else:
                        break
                
                if same_count >= 8:
                    return True

        return False

    def remove_repetitive_patterns(self, text: str) -> str:
        """반복 패턴 제거"""
        if not text:
            return ""

        # 문제가 되는 특정 패턴만 제거
        problematic_removals = [
            "갈취 묻는 말",
            "묻고 갈취",
        ]

        for pattern in problematic_removals:
            text = text.replace(pattern, "")

        # 연속된 동일 단어를 3개까지 허용
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            count = 1
            while i + count < len(words) and words[i + count] == current_word:
                count += 1

            # 최대 3개까지 허용
            if count >= 5:  # 5개 이상만 제한
                cleaned_words.extend([current_word] * min(3, count))
            else:
                cleaned_words.extend([current_word] * count)

            i += count

        text = " ".join(cleaned_words)

        # 반복되는 구문 패턴 제거
        text = re.sub(r"(.{5,15})\s*\1\s*\1\s*\1+", r"\1", text)  # 4회 이상만 제거

        # 불필요한 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def recover_korean_text(self, text: str) -> str:
        """한국어 텍스트 복구"""
        if not text:
            return ""

        # 치명적인 반복 패턴만 제거
        if self.detect_critical_repetitive_patterns(text):
            text = self.remove_repetitive_patterns(text)

        # 유니코드 정규화
        text = unicodedata.normalize("NFC", text)

        # 깨진 문자 복구
        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        # 품질 패턴 적용
        for pattern_config in self.korean_quality_patterns:
            pattern = pattern_config["pattern"]
            replacement = pattern_config["replacement"]
            text = re.sub(pattern, replacement, text)

        # 추가 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def enhance_korean_answer_quality(
        self, answer: str, question: str = "", intent_analysis: Dict = None
    ) -> str:
        """한국어 답변 품질 향상"""
        if not answer:
            return ""

        # 치명적인 반복 패턴만 조기 감지
        if self.detect_critical_repetitive_patterns(answer):
            answer = self.remove_repetitive_patterns(answer)
            # 최소 길이 기준 완화
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        # 기본 복구
        answer = self.recover_korean_text(answer)

        # 의도별 개선
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")

            # 기관명 답변 개선
            if answer_type == "기관명":
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

        # 문법 및 구조 개선
        if len(answer) > 10 and not answer.endswith((".", "다", "요", "함")):
            if answer.endswith("니"):
                answer += "다."
            elif answer.endswith("습"):
                answer += "니다."
            else:
                answer += "."

        # 길이 조절
        if len(answer) > 500:
            sentences = answer.split(". ")
            if len(sentences) > 4:
                answer = ". ".join(sentences[:4])
                if not answer.endswith("."):
                    answer += "."

        # 최종 정리
        answer = re.sub(r"\s+", " ", answer).strip()

        # 최종 검증
        if self.detect_critical_repetitive_patterns(answer):
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립해야 합니다."

        return answer

    def _generate_safe_fallback_answer(self, intent_type: str) -> str:
        """안전한 폴백 답변"""
        fallback_templates = {
            "기관_묻기": "관련 전문 기관에서 해당 업무를 담당하고 있습니다.",
            "특징_묻기": "주요 특징을 체계적으로 분석하여 관리해야 합니다.",
            "지표_묻기": "주요 탐지 지표를 통해 모니터링과 분석을 수행해야 합니다.",
            "방안_묻기": "체계적인 대응 방안을 수립하고 실행해야 합니다.",
            "절차_묻기": "관련 절차에 따라 단계별로 수행해야 합니다.",
            "조치_묻기": "적절한 보안 조치를 시행해야 합니다.",
        }
        
        return fallback_templates.get(intent_type, "관련 법령과 규정에 따라 체계적인 관리가 필요합니다.")

    def _extract_choice_count(self, question: str) -> int:
        """선택지 개수 추출"""
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

        # 폴백 패턴 확인
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
        """한국어 프롬프트 생성"""
        domain = self._detect_domain(question)

        # 기본 한국어 전용 지시
        korean_instruction = """
반드시 다음 규칙을 엄격히 준수하여 답변하세요:
1. 완전한 한국어로만 답변 작성
2. 영어나 외국어 절대 사용 금지
3. 깨진 문자나 특수 기호 사용 금지
4. 완전한 한국어 문장으로 구성
5. 자연스러운 한국어 표현 사용
6. 의미 있는 전문적 내용으로 구성
7. 논리적이고 일관된 설명
8. 완전하고 명확한 문장으로 마무리
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

            # 템플릿 예시 추가
            if domain_hints and "template_examples" in domain_hints:
                examples = domain_hints["template_examples"]
                if examples and isinstance(examples, list) and len(examples) > 0:
                    # 모든 예시 활용
                    template_examples = "\n\n참고 예시 (이와 유사한 수준과 구조로 작성하세요):\n"
                    for i, example in enumerate(examples[:5], 1):  # 최대 5개
                        template_examples += f"참고 {i}: {example}\n"

                    template_examples += "\n위 예시들을 참고하여 질문에 맞는 구체적이고 전문적인 답변을 작성하세요."

            # 답변 유형별 추가 지침
            if answer_type == "기관명":
                intent_instruction += "\n구체적인 기관명과 소속을 정확한 한국어로 명시하세요."
            elif answer_type == "특징설명":
                intent_instruction += "\n주요 특징을 체계적으로 한국어로 나열하고 상세히 설명하세요."
            elif answer_type == "지표나열":
                intent_instruction += "\n탐지 지표를 구체적으로 한국어로 설명하고 실무적 관점에서 제시하세요."
            elif answer_type == "방안제시":
                intent_instruction += "\n실무적 대응방안을 단계별로 한국어로 제시하세요."

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
                    hint_context += "\n완전한 한국어로만 작성하세요."
                elif improvement_type == "intent_mismatch":
                    hint_context += f"\n질문 의도에 정확히 부합하는 답변을 작성하세요."

        if question_type == "multiple_choice":
            return self._create_enhanced_mc_prompt(
                question, self._extract_choice_count(question), domain, domain_hints
            )
        else:
            # 주관식 프롬프트
            prompt_templates = [
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
- 실무적이고 구체적인 내용 포함
- 전문적 수준의 상세한 설명

답변:""",
            ]

            return prompt_templates[0]

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
        """답변 생성"""

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

            # knowledge_base에서 템플릿 예시 가져오기
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
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500,
                add_special_tokens=True,
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            # 생성 설정
            gen_config = self._get_generation_config(question_type)
            
            # 주관식의 경우 더 관대한 설정
            if question_type == "subjective":
                gen_config.repetition_penalty = 1.1
                gen_config.no_repeat_ngram_size = 2
                gen_config.temperature = 0.6
                gen_config.top_p = 0.9

            # 모델 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # 디코딩
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            # 치명적인 반복 패턴만 조기 감지
            if self.detect_critical_repetitive_patterns(response):
                return self._retry_generation_with_different_settings(
                    prompt, question_type, max_choice, intent_analysis
                )

            # 후처리
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
            
            # 폴백에서도 의도 기반 답변 시도
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
        """지식베이스에서 템플릿 예시 가져오기"""
        templates_mapping = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다.",
                    "트로이 목마는 유용한 프로그램으로 가장하여 사용자가 직접 설치하도록 유도하고, 설치 후 악의적인 기능을 수행하는 특징을 가집니다. 원격 접근 기능을 통해 시스템을 외부에서 조작할 수 있습니다.",
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다.",
                    "시스템 성능 저하, 예상치 못한 네트워크 활동, 방화벽 로그의 이상 패턴, 파일 시스템 변경 사항, 사용자 계정의 비정상적 활동 등이 주요 탐지 지표로 활용됩니다.",
                ],
                "방안_묻기": [
                    "딥페이크 기술 악용에 대비하여 다층 방어체계 구축, 실시간 딥페이크 탐지 시스템 도입, 직원 교육 및 인식 개선, 생체인증 강화, 다중 인증 체계 구축 등의 종합적 대응방안이 필요합니다.",
                    "네트워크 분할을 통한 격리, 접근권한 최소화 원칙 적용, 행위 기반 탐지 시스템 구축, 사고 대응 절차 수립, 백업 및 복구 체계 마련 등의 보안 강화 방안을 수립해야 합니다.",
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
            "정보보안": {
                "방안_묻기": [
                    "정보보안관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다.",
                    "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다.",
                ],
            },
            "금융투자": {
                "방안_묻기": [
                    "투자자 보호를 위한 적합성 원칙 준수, 투자위험 고지 의무 이행, 투자자문 서비스 품질 개선, 불공정거래 방지 체계 구축, 내부통제 시스템 강화 등의 방안이 필요합니다.",
                ],
            },
            "위험관리": {
                "방안_묻기": [
                    "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다.",
                    "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다.",
                ],
            },
        }

        if domain in templates_mapping and intent_key in templates_mapping[domain]:
            return templates_mapping[domain][intent_key]

        # 일반 템플릿
        general_templates = {
            "특징_묻기": [
                "주요 특징을 체계적으로 분석하여 관리해야 합니다.",
                "핵심적인 특성과 성질을 정확히 파악하여 대응해야 합니다.",
            ],
            "지표_묻기": [
                "주요 탐지 지표를 통해 모니터링과 분석을 수행해야 합니다.",
                "관련 징후와 패턴을 체계적으로 분석하여 식별해야 합니다.",
            ],
            "방안_묻기": [
                "체계적인 대응 방안을 수립하고 실행해야 합니다.",
                "효과적인 관리 방안을 마련하여 지속적으로 개선해야 합니다.",
            ],
            "기관_묻기": [
                "관련 전문 기관에서 해당 업무를 담당하고 있습니다.",
            ],
        }

        return general_templates.get(intent_key, [])

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
                max_new_tokens=250 if question_type == "subjective" else 10,
                temperature=0.5,
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=retry_config)

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            # 치명적인 반복 패턴만 확인
            if self.detect_critical_repetitive_patterns(response):
                # 의도 기반 폴백
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
        """주관식 답변 처리"""
        if not response:
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        # 치명적인 반복 패턴만 조기 감지
        if self.detect_critical_repetitive_patterns(response):
            response = self.remove_repetitive_patterns(response)
            if len(response) < 20:
                if intent_analysis:
                    primary_intent = intent_analysis.get("primary_intent", "일반")
                    return self._generate_safe_fallback_answer(primary_intent)
                return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        # 한국어 텍스트 복구
        response = self.recover_korean_text(response)

        # 품질 향상
        response = self.enhance_korean_answer_quality(
            response, question, intent_analysis
        )

        # 기본 정리
        response = re.sub(r"\s+", " ", response).strip()

        # 불필요한 내용 제거
        response = re.sub(r"답변[:：]\s*", "", response)
        response = re.sub(r"질문[:：].*?\n", "", response)
        response = re.sub(r"다음.*?답변하세요[.:]\s*", "", response)

        # 한국어 검증
        korean_ratio = self._calculate_korean_ratio(response)

        # 의도별 답변 검증 및 개선
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

        # 최종 검증 및 보완
        if korean_ratio < 0.5 or len(response) < 15:
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                response = self._generate_safe_fallback_answer(primary_intent)
            else:
                response = "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        # 길이 조절
        if len(response) > 450:
            sentences = response.split(". ")
            response = ". ".join(sentences[:4])
            if not response.endswith("."):
                response += "."

        # 마침표 확인
        if (
            response
            and not response.endswith((".", "다", "요", "함"))
            and "생성에 실패" not in response
            and "관리가 필요" not in response
        ):
            response += "."

        # 최종 치명적 반복 패턴만 확인
        if self.detect_critical_repetitive_patterns(response):
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return "관련 법령과 규정에 따라 체계적인 관리가 필요합니다."

        return response

    def _process_enhanced_mc_answer(
        self, response: str, question: str, max_choice: int
    ) -> str:
        """객관식 답변 처리"""
        if max_choice <= 0:
            max_choice = 5

        # 텍스트 복구
        response = self.recover_korean_text(response)

        # 숫자 추출
        numbers = re.findall(r"[1-9]", response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
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
            # 폴백에서는 더 관대한 설정
            gen_config.repetition_penalty = 1.1
            gen_config.no_repeat_ngram_size = 2
            gen_config.temperature = 0.7

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
        """생성 설정"""
        config_dict = GENERATION_CONFIG[question_type].copy()
        config_dict["pad_token_id"] = self.tokenizer.pad_token_id
        config_dict["eos_token_id"] = self.tokenizer.eos_token_id

        # 주관식에 더 관대한 설정
        if question_type == "subjective":
            config_dict["repetition_penalty"] = 1.1
            config_dict["no_repeat_ngram_size"] = 2
            config_dict["temperature"] = 0.6
            config_dict["top_p"] = 0.9
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
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                return self._generate_safe_fallback_answer(primary_intent)
            return self.generate_fallback_subjective_answer(question)

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

    def cleanup(self):
        """리소스 정리"""
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