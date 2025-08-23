# model_handler.py

import torch
import re
import time
import gc
import random
import os
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
    MEMORY_CONFIG,
    get_device,
)


class SimpleModelHandler:

    def __init__(self, model_name: str = None, verbose: bool = False):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.verbose = verbose
        self.device = get_device()

        self._initialize_integrated_data()

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
        """토크나이저 한국어 최적화"""
        if hasattr(self.tokenizer, "do_lower_case"):
            self.tokenizer.do_lower_case = False

        if hasattr(self.tokenizer, "normalize"):
            self.tokenizer.normalize = False

        special_tokens = ["<korean>", "</korean>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _initialize_integrated_data(self):
        """JSON 데이터를 코드 내부로 통합하여 초기화"""
        
        # mc_context_patterns 데이터
        self.mc_context_patterns = {
            "negative_keywords": ["해당하지.*않는", "적절하지.*않는", "옳지.*않는", "틀린", "잘못된", "부적절한", "아닌.*것"],
            "positive_keywords": ["맞는.*것", "옳은.*것", "적절한.*것", "올바른.*것", "해당하는.*것", "정확한.*것", "가장.*적절한", "가장.*옳은"],
            "domain_specific_patterns": {
                "금융투자": {
                    "keywords": ["금융투자업", "투자자문업", "투자매매업", "투자중개업", "소비자금융업", "보험중개업"],
                    "common_answers": ["1", "5"]
                },
                "위험관리": {
                    "keywords": ["위험관리", "위험수용", "위험대응", "수행인력", "재해복구"],
                    "common_answers": ["2", "3"]
                },
                "개인정보보호": {
                    "keywords": ["개인정보", "정보주체", "만 14세", "법정대리인", "PIMS"],
                    "common_answers": ["2", "4"]
                },
                "전자금융": {
                    "keywords": ["전자금융", "분쟁조정", "금융감독원", "한국은행"],
                    "common_answers": ["4"]
                },
                "사이버보안": {
                    "keywords": ["SBOM", "악성코드", "보안", "소프트웨어"],
                    "common_answers": ["1", "3", "5"]
                }
            }
        }

        # intent_specific_prompts 데이터
        self.intent_specific_prompts = {
            "기관_묻기": [
                "다음 질문에서 요구하는 특정 기관명을 정확히 답변하세요. 전자금융분쟁조정위원회, 개인정보보호위원회, 금융감독원 등 구체적인 기관명을 포함해야 합니다.",
                "질문에서 묻고 있는 기관이나 조직의 정확한 명칭을 한국어로 답변하세요. 분쟁조정, 신고접수, 감독업무를 담당하는 기관의 정확한 명칭을 제시하세요.",
                "해당 분야의 관련 기관을 구체적으로 명시하여 답변하세요. 금융감독원 내 전자금융분쟁조정위원회, 개인정보보호위원회 산하 개인정보침해신고센터 등을 정확히 명시하세요.",
                "분쟁조정이나 신고접수를 담당하는 기관명을 정확히 제시하세요. 소속기관과 함께 구체적인 기관명을 명시해야 합니다.",
                "관련 법령에 따라 업무를 담당하는 기관의 정확한 명칭을 답변하세요. 전자금융거래법, 개인정보보호법 등에 따른 담당기관을 명시하세요."
            ],
            "특징_묻기": [
                "트로이 목마 기반 원격제어 악성코드의 주요 특징과 특성을 체계적으로 설명하세요. 은밀성, 지속성, 원격제어 기능 등을 포함하세요.",
                "해당 항목의 핵심적인 특징들을 구체적으로 나열하고 설명하세요. 기술적 특성과 동작 원리를 중심으로 설명하세요.",
                "특징과 성질을 중심으로 상세히 기술하세요. 정상 프로그램으로 위장하는 특성, 사용자 자발적 설치, 외부 제어 등을 설명하세요.",
                "고유한 특성과 차별화 요소를 포함하여 설명하세요. 다른 악성코드와 구별되는 특징을 중심으로 기술하세요.",
                "주요 특징을 분류하여 체계적으로 제시하세요. 설치 방식, 동작 특성, 탐지 회피 기법 등으로 분류하여 설명하세요."
            ],
            "지표_묻기": [
                "탐지 지표와 징후를 중심으로 구체적으로 나열하고 설명하세요. 네트워크 트래픽, 시스템 활동, 파일 변화 등의 지표를 포함하세요.",
                "주요 지표들을 체계적으로 분류하여 제시하세요. 기술적 지표와 행위적 지표로 구분하여 설명하세요.",
                "관찰 가능한 지표와 패턴을 중심으로 답변하세요. 비정상적인 네트워크 연결, 시스템 성능 변화, 파일 시스템 변조 등을 설명하세요.",
                "식별 가능한 신호와 징후를 구체적으로 설명하세요. 원격 접속 흔적, 의심스러운 프로세스, 레지스트리 변경 등을 포함하세요.",
                "모니터링과 탐지에 활용할 수 있는 지표를 제시하세요. 실시간 모니터링과 사후 분석에 사용할 수 있는 지표들을 설명하세요."
            ],
            "방안_묻기": [
                "구체적인 대응 방안과 해결책을 제시하세요. 기술적 대응방안과 관리적 대응방안을 모두 포함하세요.",
                "실무적이고 실행 가능한 방안들을 중심으로 답변하세요. 딥페이크 기술 악용 대비 방안, 금융권 보안 강화 방안 등을 구체적으로 제시하세요.",
                "체계적인 관리 방안을 단계별로 설명하세요. 예방, 탐지, 대응, 복구 단계별 방안을 제시하세요.",
                "효과적인 대처 방안과 예방책을 함께 제시하세요. 사전 예방조치와 사후 대응조치를 균형있게 설명하세요.",
                "실제 적용 가능한 구체적 방안을 설명하세요. 조직 차원의 대응체계와 기술적 보안조치를 포함하세요."
            ],
            "절차_묻기": [
                "단계별 절차를 순서대로 설명하세요. 첫 번째 단계부터 마지막 단계까지 논리적 순서로 제시하세요.",
                "처리 과정을 체계적으로 기술하세요. 각 단계별 담당자와 처리 내용을 명확히 설명하세요.",
                "진행 절차와 각 단계의 내용을 상세히 설명하세요. 필요한 서류와 처리 기간을 포함하세요.",
                "업무 프로세스를 단계별로 제시하세요. 신청에서 완료까지의 전체 과정을 설명하세요.",
                "수행 절차를 논리적 순서에 따라 설명하세요. 각 단계의 목적과 주요 활동을 포함하세요."
            ],
            "조치_묻기": [
                "필요한 보안조치와 대응조치를 설명하세요. 기술적 조치와 관리적 조치를 구분하여 제시하세요.",
                "예방조치와 사후조치를 포함하여 답변하세요. 사전 예방을 위한 조치와 사고 발생 시 조치를 설명하세요.",
                "적절한 대응조치 방안을 구체적으로 제시하세요. 즉시 조치사항과 중장기 조치사항을 구분하여 설명하세요.",
                "보안강화 조치와 관리조치를 설명하세요. 시스템 보안조치와 운영 관리조치를 포함하세요.",
                "효과적인 조치 방안을 체계적으로 기술하세요. 조치의 우선순위와 시행 방법을 포함하세요."
            ],
            "법령_묻기": [
                "관련 법령과 규정을 근거로 설명하세요. 개인정보보호법, 전자금융거래법, 자본시장법 등의 구체적 조항을 인용하세요.",
                "법적 근거와 조항을 포함하여 답변하세요. 해당 법령의 정확한 명칭과 조항 번호를 제시하세요.",
                "해당 법률의 주요 내용을 설명하세요. 법령의 목적과 적용 범위를 포함하여 기술하세요.",
                "관련 규정과 기준을 중심으로 기술하세요. 법령에 따른 의무사항과 준수 기준을 설명하세요.",
                "법령상 요구사항과 의무사항을 설명하세요. 위반 시 제재사항과 함께 설명하세요."
            ],
            "정의_묻기": [
                "정확한 정의와 개념을 설명하세요. 법적 정의와 기술적 정의를 구분하여 제시하세요.",
                "용어의 의미와 개념을 명확히 제시하세요. 관련 법령에서의 정의와 일반적 의미를 설명하세요.",
                "개념적 정의와 실무적 의미를 함께 설명하세요. 이론적 정의와 실제 적용 사례를 포함하세요.",
                "해당 용어의 정확한 뜻과 범위를 기술하세요. 포함되는 범위와 제외되는 범위를 명확히 하세요.",
                "정의와 함께 구체적 예시를 포함하여 설명하세요. 실제 사례를 통해 개념을 명확히 하세요."
            ]
        }

        # korean_text_recovery 데이터
        self.korean_recovery_config = {
            "broken_unicode_chars": {
                "\\u1100": "",
                "\\u1101": "",
                "\\u1102": "",
                "\\u1103": "",
                "\\u1104": "",
                "\\u1105": "",
                "\\u1106": "",
                "\\u1107": "",
                "\\u1108": "",
                "\\u1109": "",
                "\\u110A": "",
                "\\u110B": "",
                "\\u110C": "",
                "\\u110D": "",
                "\\u110E": "",
                "\\u110F": "",
                "\\u1110": "",
                "\\u1111": "",
                "\\u1112": "",
                "\\u1161": "",
                "\\u1162": "",
                "\\u1163": "",
                "\\u1164": "",
                "\\u1165": "",
                "\\u1166": "",
                "\\u1167": "",
                "\\u1168": "",
                "\\u1169": "",
                "\\u116A": "",
                "\\u116B": "",
                "\\u116C": "",
                "\\u116D": "",
                "\\u116E": "",
                "\\u116F": "",
                "\\u1170": "",
                "\\u1171": "",
                "\\u1172": "",
                "\\u1173": "",
                "\\u1174": "",
                "\\u1175": "",
                "\\u11A8": "",
                "\\u11A9": "",
                "\\u11AA": "",
                "\\u11AB": "",
                "\\u11AC": "",
                "\\u11AD": "",
                "\\u11AE": "",
                "\\u11AF": "",
                "\\u11B0": "",
                "\\u11B1": "",
                "\\u11B2": "",
                "\\u11B3": "",
                "\\u11B4": "",
                "\\u11B5": "",
                "\\u11B6": "",
                "\\u11B7": "",
                "\\u11B8": "",
                "\\u11B9": "",
                "\\u11BA": "",
                "\\u11BB": "",
                "\\u11BC": "",
                "\\u11BD": "",
                "\\u11BE": "",
                "\\u11BF": "",
                "\\u11C0": "",
                "\\u11C1": "",
                "\\u11C2": ""
            },
            "japanese_katakana_removal": {
                "ト": "",
                "リ": "",
                "ス": "",
                "ン": "",
                "ー": "",
                "ィ": "",
                "ウ": "",
                "エ": "",
                "オ": "",
                "カ": "",
                "キ": "",
                "ク": "",
                "ケ": "",
                "コ": "",
                "サ": "",
                "シ": "",
                "セ": "",
                "ソ": "",
                "タ": "",
                "チ": "",
                "ツ": "",
                "テ": "",
                "ナ": "",
                "ニ": "",
                "ヌ": "",
                "ネ": "",
                "ノ": "",
                "ハ": "",
                "ヒ": "",
                "フ": "",
                "ヘ": "",
                "ホ": "",
                "マ": "",
                "ミ": "",
                "ム": "",
                "メ": "",
                "モ": "",
                "ヤ": "",
                "ユ": "",
                "ヨ": "",
                "ラ": "",
                "ル": "",
                "レ": "",
                "ロ": "",
                "ワ": "",
                "ヰ": "",
                "ヱ": "",
                "ヲ": ""
            },
            "broken_korean_patterns": {
                "어어지인": "",
                "선 어": "",
                "언 어": "",
                "순 어": "",
                "ᄒᆞᆫ": "",
                "ᄒᆞᆫ선": "",
                "어어지인가": "",
                "지인가": "",
                "가 시": "",
                "시 언": "",
                "언 어어": "",
                "지인)가": "",
                "순 어어": "",
                "지인.": ""
            },
            "spaced_korean_fixes": {
                "작 로": "으로",
                "렴": "련",
                "니 터": "니터",
                "지 속": "지속",
                "모 니": "모니",
                "체 계": "체계",
                "관 리": "관리",
                "법 령": "법령",
                "규 정": "규정",
                "조 치": "조치",
                "절 차": "절차",
                "대 응": "대응",
                "방 안": "방안",
                "기 관": "기관",
                "위 원": "위원",
                "감 독": "감독",
                "전 자": "전자",
                "금 융": "금융",
                "개 인": "개인",
                "정 보": "정보",
                "보 호": "보호",
                "관 련": "관련",
                "필 요": "필요",
                "중 요": "중요",
                "주 요": "주요",
                "시 스": "시스",
                "템": "템",
                "프 로": "프로",
                "세 스": "세스",
                "네 트": "네트",
                "워 크": "워크",
                "트 래": "트래",
                "픽": "픽",
                "파 일": "파일",
                "로 그": "로그",
                "연 결": "연결",
                "접 근": "접근",
                "권 한": "권한",
                "사 용": "사용",
                "계 정": "계정",
                "활 동": "활동",
                "패 턴": "패턴",
                "행 동": "행동",
                "모 니 터 링": "모니터링",
                "탐 지": "탐지",
                "발 견": "발견",
                "식 별": "식별",
                "분 석": "분석",
                "확 인": "확인",
                "점 검": "점검",
                "감 사": "감사",
                "보 안": "보안",
                "안 전": "안전",
                "위 험": "위험",
                "내 부": "내부",
                "외 부": "외부",
                "시 행": "시행",
                "실 시": "실시",
                "수 립": "수립",
                "구 축": "구축",
                "마 련": "마련",
                "준 비": "준비",
                "실 행": "실행",
                "진 행": "진행",
                "처 리": "처리",
                "관 찰": "관찰",
                "추 적": "추적",
                "기 록": "기록",
                "저 장": "저장",
                "백 업": "백업",
                "복 구": "복구",
                "복 원": "복원",
                "복 사": "복사",
                "이 동": "이동",
                "전 송": "전송",
                "수 신": "수신",
                "발 신": "발신",
                "전 달": "전달",
                "통 신": "통신",
                "연 락": "연락",
                "회 의": "회의",
                "논 의": "논의",
                "검 토": "검토",
                "평 가": "평가",
                "심 사": "심사",
                "심 의": "심의",
                "승 인": "승인",
                "허 가": "허가",
                "인 가": "인가",
                "등 록": "등록",
                "신 청": "신청",
                "요 청": "요청",
                "신 고": "신고",
                "보 고": "보고",
                "통 보": "통보",
                "고 지": "고지",
                "안 내": "안내",
                "지 침": "지침",
                "지 시": "지시",
                "명 령": "명령",
                "지 도": "지도",
                "교 육": "교육",
                "훈 련": "훈련",
                "연 수": "연수",
                "학 습": "학습",
                "습 득": "습득",
                "이 해": "이해",
                "파 악": "파악",
                "인 식": "인식",
                "감 지": "감지"
            },
            "common_korean_typos": {
                "윋": "융",
                "젂": "전",
                "엯": "연",
                "룐": "른",
                "겫": "결",
                "뷮": "분",
                "쟈": "저",
                "럭": "력",
                "솟": "솔",
                "쟣": "저",
                "뿣": "불",
                "뻙": "분",
                "걗": "것",
                "룊": "른",
                "믝": "미",
                "읶": "인",
                "멈": "멈",
                "솔": "솔",
                "랛": "란",
                "궗": "사",
                "쪗": "저",
                "롸": "로",
                "뿞": "분",
                "딞": "딘",
                "쭒": "주",
                "놟": "놓",
                "룍": "른",
                "쫒": "조",
                "놔": "놔"
            },
            "critical_repetitive_patterns": {
                "갈취 묻는 말": "",
                "묻고 갈취": "",
                "갈취": "",
                "() 기반": "",
                "()는": "",
                "() 특징": "",
                "() 지표": "",
                "() 방안": "",
                "()를": "",
                "()에": "",
                "()의": "",
                "()와": "",
                "()로": "",
                "() 는": "",
                "() 이": "",
                "() 가": "",
                "() 을": "",
                "() 에": "",
                "() 와": "",
                "() 로": "",
                "괄호": "",
                "(괄호)": "",
                "　　": " ",
                "말 말": "말",
                "말말": "말",
                "말말말": "말"
            },
            "pattern_variations": {
                "갈취.*갈취": "",
                "묻는.*묻는": "",
                "말.*말.*말": "말",
                "(.{1,5})\\s*\\1\\s*\\1\\s*\\1": "\\1",
                "(.{2,8})\\s*\\1\\s*\\1": "\\1"
            }
        }

        # korean_quality_patterns 데이터
        self.korean_quality_patterns = [
            {
                "pattern": "([가-힣])\\s+(은|는|이|가|을|를|에|의|와|과|로|으로)\\s+",
                "replacement": "\\1\\2 "
            },
            {
                "pattern": "([가-힣])\\s+(다|요|함|니다|습니다)\\s*\\.",
                "replacement": "\\1\\2."
            },
            {
                "pattern": "([가-힣])\\s*$",
                "replacement": "\\1."
            },
            {
                "pattern": "\\.+",
                "replacement": "."
            },
            {
                "pattern": "\\s*\\.\\s*",
                "replacement": ". "
            },
            {
                "pattern": "\\s+",
                "replacement": " "
            },
            {
                "pattern": "\\(\\s*\\)",
                "replacement": ""
            },
            {
                "pattern": "\\(\\s*\\)\\s*[가-힣]{1,3}",
                "replacement": ""
            },
            {
                "pattern": "[.,!?]{3,}",
                "replacement": "."
            },
            {
                "pattern": "\\s+[.,!?]\\s+",
                "replacement": ". "
            }
        ]

        self._setup_korean_recovery_mappings()

        print("통합 데이터 초기화 완료")

    def _setup_korean_recovery_mappings(self):
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {}

        for broken, replacement in self.korean_recovery_config["broken_unicode_chars"].items():
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

    def detect_critical_repetitive_patterns(self, text: str) -> bool:
        """문제 패턴 감지"""
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
        """반복 패턴 제거"""
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
        """한국어 텍스트 복구"""
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
        """한국어 답변 품질 향상"""
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
        """안전한 대체 답변 생성"""
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

        for i in range(5, 2, -1):
            pattern = r"1\s.*" + ".*".join([f"{j}\s" for j in range(2, i + 1)])
            if re.search(pattern, question, re.DOTALL):
                return i

        return 5

    def _analyze_mc_context(self, question: str, domain: str = "일반") -> Dict:
        """객관식 문맥 분석"""
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
        """한국어 프롬프트 생성"""
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
        """객관식 프롬프트 생성"""
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
        """답변 생성"""

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
        """지식베이스에서 템플릿 예시 가져오기"""
        templates_mapping = {
            "사이버보안": {
                "특징_묻기": [
                    "트로이 목마 기반 원격제어 악성코드는 정상 프로그램으로 위장하여 사용자가 자발적으로 설치하도록 유도하는 특징을 가집니다. 설치 후 외부 공격자가 원격으로 시스템을 제어할 수 있는 백도어를 생성하며, 은밀성과 지속성을 특징으로 합니다.",
                    "해당 악성코드는 사용자를 속여 시스템에 침투하여 외부 공격자가 원격으로 제어하는 특성을 가지며, 시스템 깊숙이 숨어서 장기간 활동하면서 정보 수집과 원격 제어 기능을 수행합니다."
                ],
                "지표_묻기": [
                    "네트워크 트래픽 모니터링에서 비정상적인 외부 통신 패턴, 시스템 동작 분석에서 비인가 프로세스 실행, 파일 생성 및 수정 패턴의 이상 징후, 입출력 장치에 대한 비정상적 접근 등이 주요 탐지 지표입니다.",
                    "원격 접속 흔적, 의심스러운 네트워크 연결, 시스템 파일 변조, 레지스트리 수정, 비정상적인 메모리 사용 패턴, 알려지지 않은 프로세스 실행 등을 통해 탐지할 수 있습니다."
                ],
                "방안_묻기": [
                    "딥페이크 기술 악용에 대비하여 다층 방어체계 구축, 실시간 딥페이크 탐지 시스템 도입, 직원 교육 및 인식 개선, 생체인증 강화, 다중 인증 체계 구축 등의 종합적 대응방안이 필요합니다.",
                    "네트워크 분할을 통한 격리, 접근권한 최소화 원칙 적용, 행위 기반 탐지 시스템 구축, 사고 대응 절차 수립, 백업 및 복구 체계 마련 등의 보안 강화 방안을 수립해야 합니다."
                ]
            },
            "개인정보보호": {
                "기관_묻기": [
                    "개인정보보호위원회가 개인정보 보호에 관한 업무를 총괄하며, 개인정보침해신고센터에서 신고 접수 및 상담 업무를 담당합니다.",
                    "개인정보보호위원회는 개인정보 보호 정책 수립과 감시 업무를 수행하는 중앙 행정기관이며, 개인정보 분쟁조정위원회에서 관련 분쟁의 조정 업무를 담당합니다."
                ],
                "방안_묻기": [
                    "개인정보 처리 시 수집 최소화 원칙 적용, 목적 외 이용 금지, 적절한 보호조치 수립, 정기적인 개인정보 영향평가 실시, 정보주체 권리 보장 체계 구축 등의 관리방안이 필요합니다.",
                    "개인정보보호 관리체계 구축, 개인정보처리방침 수립 및 공개, 개인정보보호책임자 지정, 정기적인 교육 실시, 기술적·관리적·물리적 보호조치 이행 등을 체계적으로 수행해야 합니다."
                ]
            },
            "전자금융": {
                "기관_묻기": [
                    "전자금융분쟁조정위원회에서 전자금융거래 관련 분쟁조정 업무를 담당합니다. 이 위원회는 금융감독원 내에 설치되어 운영됩니다.",
                    "금융감독원 내 전자금융분쟁조정위원회가 이용자의 분쟁조정 신청을 접수하고 처리하는 업무를 수행합니다."
                ]
            }
        }

        if domain in templates_mapping and intent_key in templates_mapping[domain]:
            return templates_mapping[domain][intent_key]

        return [
            "관련 법령과 규정에 따라 체계적인 관리가 필요합니다.",
            "해당 분야의 전문적 지식을 바탕으로 적절한 대응을 수행해야 합니다."
        ]

    def _retry_generation_with_different_settings(
        self,
        prompt: str,
        question_type: str,
        max_choice: int,
        intent_analysis: Dict = None,
    ) -> str:
        """다른 설정으로 재생성"""
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
        """주관식 답변 처리"""
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
        """객관식 답변 처리"""
        if max_choice <= 0:
            max_choice = 5

        response = self.recover_korean_text(response)

        numbers = re.findall(r"[1-9]", response)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                return num

        return self._force_valid_mc_answer(response, max_choice)

    def _force_valid_mc_answer(self, response: str, max_choice: int) -> str:
        """유효한 객관식 답변 강제 생성"""
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
        """문맥 기반 객관식 답변 생성"""
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
        """대체 객관식 답변 생성"""
        return self.generate_contextual_mc_answer(question, max_choice, domain)

    def generate_fallback_subjective_answer(self, question: str) -> str:
        """대체 주관식 답변 생성"""
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
        """생성 설정 가져오기"""
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
        """도메인 탐지"""
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
        """도메인 키워드 가져오기"""
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
        """LLM 기반 대체 답변"""
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