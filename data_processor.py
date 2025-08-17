# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
- 질문 의도 분석
- LLM 생성 결과 후처리 강화
"""

import re
import pickle
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from config import (
    PKL_DIR, KOREAN_REQUIREMENTS, JSON_CONFIG_FILES, 
    TEXT_CLEANUP_CONFIG, KOREAN_TYPO_MAPPING, OPTIMIZATION_CONFIG,
    TEXT_SAFETY_CONFIG, check_text_safety
)

class SimpleDataProcessor:
    """데이터 처리기 - LLM 생성 결과 후처리 중심"""
    
    def __init__(self):
        self.pkl_dir = PKL_DIR
        self.pkl_dir.mkdir(exist_ok=True)
        
        self._load_json_configs()
        
        self.korean_requirements = KOREAN_REQUIREMENTS
        self.text_cleanup_config = TEXT_CLEANUP_CONFIG
        self.korean_typo_mapping = KOREAN_TYPO_MAPPING
        self.text_safety_config = TEXT_SAFETY_CONFIG
        
        self.processing_stats = self.processing_stats_structure.copy()
        
        self._load_processing_history()
    
    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            with open(JSON_CONFIG_FILES['processing_config'], 'r', encoding='utf-8') as f:
                processing_config = json.load(f)
            
            self.mc_patterns = processing_config['mc_patterns']
            self.mc_keywords = processing_config['mc_keywords']
            self.question_intent_patterns = processing_config['question_intent_patterns']
            self.subj_patterns = processing_config['subj_patterns']
            self.processing_stats_structure = processing_config['processing_stats_structure']
            self.domain_detection_patterns = processing_config.get('domain_detection_patterns', {})
            self.choice_extraction_patterns = processing_config.get('choice_extraction_patterns', [])
            self.answer_validation_rules = processing_config.get('answer_validation_rules', {})
            
            with open(JSON_CONFIG_FILES['knowledge_data'], 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            self.domain_keywords = knowledge_data['domain_keywords']
            
            print("데이터 처리 설정 파일 로드 완료")
            
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
        """기본 설정 로드"""
        print("기본 설정으로 대체합니다.")
        
        self.mc_patterns = [
            r'1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w]',
            r'①.*②.*③.*④.*⑤'
        ]
        
        self.mc_keywords = [
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것',
            r'옳지.*않는.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것'
        ]
        
        self.question_intent_patterns = {
            "기관_묻기": ["기관.*기술하세요", "기관.*설명하세요"],
            "특징_묻기": ["특징.*설명하세요", "특징.*기술하세요"],
            "지표_묻기": ["지표.*설명하세요", "탐지.*지표"]
        }
        
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요',
            r'서술하세요'
        ]
        
        self.domain_keywords = {
            "사이버보안": ["트로이", "RAT", "원격제어", "악성코드"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원"],
            "일반": ["법령", "규정", "관리", "조치"]
        }
        
        self.domain_detection_patterns = {}
        self.choice_extraction_patterns = []
        self.answer_validation_rules = {}
        
        self.processing_stats_structure = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0},
            "llm_generation_enhanced": 0,
            "text_post_processing": 0
        }
    
    def _load_processing_history(self):
        """이전 처리 기록 로드"""
        history_file = self.pkl_dir / "processing_history.pkl"
        
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    saved_stats = pickle.load(f)
                    self.processing_stats.update(saved_stats)
            except Exception:
                pass
    
    def _save_processing_history(self):
        """처리 기록 저장"""
        history_file = self.pkl_dir / "processing_history.pkl"
        
        try:
            save_data = {
                **self.processing_stats,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def detect_text_corruption(self, text: str) -> bool:
        """텍스트 깨짐 감지"""
        if not text:
            return True
        
        # config.py의 안전성 검사 활용
        return not check_text_safety(text)
    
    def clean_korean_text_premium(self, text: str) -> str:
        """프리미엄 한국어 텍스트 정리 - LLM 생성 결과 후처리"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 1단계: 깨진 텍스트 감지
        if self.detect_text_corruption(text):
            return ""
        
        # 2단계: LLM 생성 특화 후처리
        text = self._post_process_llm_output(text)
        
        # 3단계: 안전한 한국어 오타 수정
        text = self._apply_korean_corrections(text)
        
        # 4단계: 구조적 정리
        text = self._structural_cleanup(text)
        
        # 5단계: 품질 향상
        text = self._enhance_text_quality(text)
        
        self.processing_stats["text_post_processing"] += 1
        
        return text
    
    def _post_process_llm_output(self, text: str) -> str:
        """LLM 출력 특화 후처리"""
        
        # LLM 생성 시 자주 나타나는 패턴 정리
        
        # 1. 반복되는 문구 제거
        text = re.sub(r'(.{10,}?)\1+', r'\1', text)
        
        # 2. 불완전한 문장 끝 정리
        text = re.sub(r'([가-힣])\s*\.{2,}', r'\1.', text)
        
        # 3. 과도한 강조 표현 정리
        text = re.sub(r'매우\s+매우', '매우', text)
        text = re.sub(r'반드시\s+반드시', '반드시', text)
        text = re.sub(r'중요한\s+중요한', '중요한', text)
        
        # 4. 불필요한 접속사 정리
        text = re.sub(r'또한\s+또한', '또한', text)
        text = re.sub(r'그리고\s+그리고', '그리고', text)
        
        # 5. 과도한 공백 정리
        text = re.sub(r'\s{3,}', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text
    
    def _apply_korean_corrections(self, text: str) -> str:
        """한국어 오타 수정 적용"""
        
        # 기본 오타 수정
        corrections = {
            "전자금윋": "전자금융",
            "캉터": "컴퓨터", 
            "하웨어": "하드웨어",
            "네됴크": "네트워크",
            "메세지": "메시지",
            "트래픁": "트래픽",
            "보안조최": "보안조치",
            "관리방안": "관리 방안",
            "데이타": "데이터",
            "시스탬": "시스템",
            "프로그럼": "프로그램",
            "악성코드들": "악성코드"
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # 띄어쓰기 개선
        spacing_fixes = [
            (r'([가-힣])조치', r'\1 조치'),
            (r'([가-힣])방안', r'\1 방안'),
            (r'([가-힣])체계', r'\1 체계'),
            (r'([가-힣])기관', r'\1 기관'),
            (r'보안([가-힣])', r'보안 \1'),
            (r'관리([가-힣])', r'관리 \1'),
            (r'시스템([가-힣])', r'시스템 \1')
        ]
        
        for pattern, replacement in spacing_fixes:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _structural_cleanup(self, text: str) -> str:
        """구조적 텍스트 정리"""
        
        # 1. 기본 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 2. 제어 문자 제거
        text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
        
        # 3. 문장 부호 정리
        text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
        text = re.sub(r'([,;:])\s*([,;:])+', r'\1', text)
        
        # 4. 괄호 정리
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        
        # 5. 불필요한 기호 제거
        text = re.sub(r'[●◎○▪▫■□◆◇△▽]', '', text)
        
        return text
    
    def _enhance_text_quality(self, text: str) -> str:
        """텍스트 품질 향상"""
        
        # 1. 문장 끝 정리
        if text and not text.endswith(('.', '다', '요', '함', '니다', '습니다')):
            if len(text) > 15:
                # 마지막 문자가 한글이면 마침표 추가
                if re.search(r'[가-힣]$', text):
                    text += '.'
        
        # 2. 전문용어 표준화
        term_standardization = {
            "정보보안관리체계": "정보보안 관리체계",
            "개인정보보호법": "개인정보보호법",
            "전자금융거래법": "전자금융거래법",
            "사이버보안": "사이버보안",
            "악성코드": "악성코드"
        }
        
        for original, standardized in term_standardization.items():
            if original != standardized:
                text = text.replace(original, standardized)
        
        # 3. 문체 통일 (다 -> 습니다 변환 등은 제한적으로 적용)
        if text.count('다.') > text.count('습니다.') and text.count('다.') > 2:
            # 전문적인 문체로 통일이 필요한 경우에만 적용
            pass
        
        return text
    
    def analyze_question_intent(self, question: str) -> Dict:
        """강화된 질문 의도 분석 - LLM 생성 지원용"""
        question_lower = question.lower()
        
        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "llm_guidance": {}
        }
        
        intent_scores = {}
        
        # 기관 관련 질문 우선 처리
        institution_patterns = [
            r"기관.*기술하세요", r"기관.*설명하세요",
            r"어떤.*기관", r"어느.*기관", r"기관.*무엇",
            r"분쟁.*조정.*기관", r"신청.*수.*있는.*기관",
            r"담당.*기관", r"관리.*기관"
        ]
        
        institution_score = 0
        for pattern in institution_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                institution_score += 2
        
        # 기관 키워드 추가 점수
        institution_keywords = [
            "전자금융", "분쟁조정", "개인정보", "침해신고", 
            "금융감독원", "개인정보보호위원회"
        ]
        
        for keyword in institution_keywords:
            if keyword in question_lower:
                institution_score += 1
        
        if institution_score >= 2:
            intent_scores["기관_묻기"] = {
                "score": institution_score,
                "patterns": ["institution_detection"],
                "llm_guidance": {
                    "structure": "구체적인 기관명과 역할을 명시하세요",
                    "content": "법적 근거와 주요 업무를 포함하세요",
                    "format": "기관명 - 역할 - 법적근거 순으로 작성"
                }
            }
        
        # RAT/트로이 관련 질문 처리
        rat_patterns = [
            r"rat.*특징", r"트로이.*특징", r"원격.*제어.*특징",
            r"rat.*지표", r"트로이.*지표", r"원격.*제어.*지표",
            r"악성코드.*특징", r"악성코드.*지표"
        ]
        
        rat_score = 0
        for pattern in rat_patterns:
            if re.search(pattern, question_lower):
                rat_score += 2
        
        if rat_score > 0:
            if any(word in question_lower for word in ["특징", "성격"]):
                intent_scores["특징_묻기"] = {
                    "score": rat_score + 1, 
                    "patterns": ["rat_feature"],
                    "llm_guidance": {
                        "structure": "악성코드의 주요 특징을 체계적으로 나열하세요",
                        "content": "원격제어 기능, 은폐 기법, 지속성을 포함하세요",
                        "format": "특징별로 구분하여 설명"
                    }
                }
            elif any(word in question_lower for word in ["지표", "징후", "탐지"]):
                intent_scores["지표_묻기"] = {
                    "score": rat_score + 1, 
                    "patterns": ["rat_indicator"],
                    "llm_guidance": {
                        "structure": "탐지 가능한 지표를 카테고리별로 설명하세요",
                        "content": "네트워크 활동, 시스템 변경, 메모리 패턴을 포함하세요",
                        "format": "지표 유형별로 구분하여 나열"
                    }
                }
        
        # 기본 의도 패턴 분석
        basic_intent_patterns = {
            "방안_묻기": {
                "patterns": [r"방안.*기술", r"대응.*방안", r"해결.*방안"],
                "llm_guidance": {
                    "structure": "실행 가능한 대응 방안을 제시하세요",
                    "content": "기술적 조치와 관리적 조치를 포함하세요",
                    "format": "방안별로 구체적인 실행 절차 제시"
                }
            },
            "절차_묻기": {
                "patterns": [r"절차.*설명", r"과정.*설명", r"단계.*설명"],
                "llm_guidance": {
                    "structure": "단계별 처리 절차를 순서대로 설명하세요",
                    "content": "각 단계의 목적과 주요 활동을 포함하세요",
                    "format": "1단계, 2단계 순으로 구조화"
                }
            },
            "조치_묻기": {
                "patterns": [r"조치.*설명", r"보안.*조치", r"대응.*조치"],
                "llm_guidance": {
                    "structure": "필요한 보안조치를 분류하여 설명하세요",
                    "content": "예방조치와 대응조치를 구분하여 제시하세요",
                    "format": "조치 유형별로 구체적인 방법 설명"
                }
            }
        }
        
        for intent_type, config in basic_intent_patterns.items():
            if intent_type in intent_scores:
                continue
                
            score = 0
            matched_patterns = []
            
            for pattern in config["patterns"]:
                if re.search(pattern, question, re.IGNORECASE):
                    score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns,
                    "llm_guidance": config["llm_guidance"]
                }
        
        # 최고 점수 의도 선택
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]
            
            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 3, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]
            intent_analysis["llm_guidance"] = best_intent[1].get("llm_guidance", {})
            
            # 답변 유형 설정
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명과 법적 근거")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("체계적인 특징 나열")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("실행 가능한 대응 방안")
            elif "절차" in primary:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["context_hints"].append("단계별 처리 절차")
            elif "조치" in primary:
                intent_analysis["answer_type_required"] = "조치설명"
                intent_analysis["context_hints"].append("보안조치와 관리조치")
        
        return intent_analysis
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """강화된 선택지 범위 추출"""
        question_type = self.analyze_question_type(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        # 개선된 선택지 추출
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            line = line.strip()
            # 다양한 패턴 지원
            patterns = [
                r'^(\d+)\s+(.+)',      # 기본: 숫자 + 공백 + 내용
                r'^(\d+)\.\s*(.+)',    # 숫자 + 점 + 공백 + 내용
                r'^(\d+)\)\s*(.+)',    # 숫자 + 괄호 + 공백 + 내용
                r'^①(.+)|^②(.+)|^③(.+)|^④(.+)|^⑤(.+)'  # 원숫자
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if pattern.startswith('^①'):
                        # 원숫자 처리
                        for i, group in enumerate(match.groups(), 1):
                            if group:
                                choice_numbers.append(i)
                                break
                    else:
                        num = int(match.group(1))
                        content = match.group(2).strip()
                        if 1 <= num <= 5 and len(content) > 0:
                            choice_numbers.append(num)
                    break
        
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            # 연속성 검증
            expected_count = max_choice - min_choice + 1
            if (len(choice_numbers) == expected_count and 
                min_choice == 1 and 
                max_choice >= 3):
                return "multiple_choice", max_choice
        
        # 폴백 - 키워드 기반 판단
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice", 5  # 기본값
        
        return "subjective", 0
    
    def analyze_question_type(self, question: str) -> str:
        """강화된 질문 유형 분석"""
        
        question = question.strip()
        
        # 주관식 패턴 우선 확인
        subjective_indicators = [
            r'설명하세요', r'기술하세요', r'서술하세요', r'작성하세요',
            r'제시하세요', r'논하시오', r'답하시오',
            r'특징.*설명', r'지표.*설명', r'방안.*기술',
            r'기관.*기술', r'절차.*설명', r'조치.*설명'
        ]
        
        for pattern in subjective_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 객관식 패턴 확인
        choice_pattern = r'\n(\d+)\s+[가-힣\w]'
        choice_matches = re.findall(choice_pattern, question)
        
        if len(choice_matches) >= 3:
            choice_nums = [int(match) for match in choice_matches]
            choice_nums.sort()
            if (choice_nums[0] == 1 and 
                len(choice_nums) == choice_nums[-1] and
                choice_nums[-1] <= 5):
                return "multiple_choice"
        
        # 원숫자 패턴 확인
        circle_pattern = r'[①②③④⑤]'
        if len(re.findall(circle_pattern, question)) >= 3:
            return "multiple_choice"
        
        # 키워드 기반 객관식 판단
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지가 실제로 있는지 확인
                if any(f'{i} ' in question for i in range(1, 6)):
                    return "multiple_choice"
        
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """강화된 도메인 추출"""
        question_lower = question.lower()
        
        domain_scores = {}
        
        # 정확한 도메인 매칭
        domain_patterns = {
            "사이버보안": ["rat", "트로이", "원격제어", "원격접근", "악성코드", "딥페이크", "sbom"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "한국은행"],
            "개인정보보호": ["개인정보", "정보주체", "만 14세", "법정대리인", "개인정보보호위원회"],
            "정보보안": ["정보보안", "isms", "관리체계", "재해복구"],
            "금융투자": ["금융투자업", "투자자문", "투자매매", "자본시장법"],
            "위험관리": ["위험관리", "위험수용", "재해복구", "위험평가"]
        }
        
        for domain, patterns in domain_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in question_lower:
                    # 핵심 키워드 가중치 적용
                    if pattern in ["rat", "트로이", "원격제어", "전자금융", "분쟁조정", "개인정보", "만 14세"]:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 통계 업데이트
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """강화된 한국어 답변 유효성 검증 - LLM 생성 결과 중심"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            # 객관식: 숫자 범위 검증
            if not (answer.isdigit() and 1 <= int(answer) <= max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
        else:
            # 주관식: LLM 생성 결과 검증
            
            # 깨진 텍스트 우선 검증
            if self.detect_text_corruption(answer):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # LLM 생성 특화 검증
            if not self._validate_llm_generated_content(answer):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 안전한 텍스트 정리
            clean_answer = self.clean_korean_text_premium(answer)
            
            if not clean_answer:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 기본 검증 기준
            min_length = 15
            max_length = 400
            min_korean_ratio = 0.6
            
            if not (min_length <= len(clean_answer) <= max_length):
                self.processing_stats["validation_failures"] += 1
                return False
            
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < min_korean_ratio:
                self.processing_stats["validation_failures"] += 1
                return False
            
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 8:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의미있는 내용 검증
            if not self._validate_meaningful_content(clean_answer, question):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            self.processing_stats["llm_generation_enhanced"] += 1
            return True
    
    def _validate_llm_generated_content(self, answer: str) -> bool:
        """LLM 생성 콘텐츠 검증"""
        
        # 1. 반복 패턴 검증
        if self._has_excessive_repetition(answer):
            return False
        
        # 2. 불완전한 문장 검증
        if self._has_incomplete_sentences(answer):
            return False
        
        # 3. 의미없는 나열 검증
        if self._has_meaningless_enumeration(answer):
            return False
        
        return True
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """과도한 반복 감지"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # 연속된 단어 반복 검사
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                consecutive_repeats += 1
                if consecutive_repeats >= 3:
                    return True
            else:
                consecutive_repeats = 0
        
        # 전체 반복 비율 검사
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repeat = max(word_counts.values())
        if max_repeat > len(words) * 0.3:  # 30% 이상 반복
            return True
        
        return False
    
    def _has_incomplete_sentences(self, text: str) -> bool:
        """불완전한 문장 감지"""
        
        # 문장이 너무 갑작스럽게 끝나는 경우
        if re.search(r'[가-힣]\s*$', text) and not text.endswith(('.', '다', '요', '함', '니다', '습니다')):
            if len(text) > 50:  # 충분히 긴 텍스트에서만 적용
                return True
        
        # 중간에 끊어진 문장
        broken_patterns = [
            r'[가-힣]이\s*$',
            r'[가-힣]가\s*$',
            r'[가-힣]를\s*$',
            r'[가-힣]의\s*$',
            r'그리고\s*$',
            r'또한\s*$',
            r'하지만\s*$'
        ]
        
        for pattern in broken_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _has_meaningless_enumeration(self, text: str) -> bool:
        """의미없는 나열 감지"""
        
        # 과도하게 짧은 문장들의 나열
        sentences = re.split(r'[.!?]', text)
        short_sentences = [s.strip() for s in sentences if len(s.strip()) < 10]
        
        if len(short_sentences) > len(sentences) * 0.5 and len(sentences) > 3:
            return True
        
        # 단순 반복 패턴
        simple_patterns = [
            r'(입니다\.)\s*\1\s*\1',
            r'(해야\s*합니다\.)\s*\1\s*\1',
            r'(필요합니다\.)\s*\1\s*\1'
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _validate_meaningful_content(self, answer: str, question: str) -> bool:
        """의미있는 내용 검증"""
        
        # 질문과 답변의 관련성 검증
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # 도메인 키워드 매칭
        found_domain_terms = 0
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    # 질문에 있는 도메인 키워드가 답변에도 있는지 확인
                    if keyword in answer_lower or any(related in answer_lower for related in keywords):
                        found_domain_terms += 1
        
        # 기본 의미 키워드 존재 확인
        meaningful_keywords = [
            "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", 
            "정책", "체계", "시스템", "통제", "대응", "처리", "기관",
            "위원회", "특징", "지표", "탐지", "모니터링", "악성코드",
            "원격제어", "트로이", "분쟁조정", "개인정보", "전자금융"
        ]
        
        keyword_count = sum(1 for word in meaningful_keywords if word in answer)
        
        # 최소 의미 기준: 도메인 관련성 또는 의미 키워드
        if found_domain_terms > 0 or keyword_count >= 1:
            return True
        
        # 길이가 충분하고 한국어 비율이 높으면 통과
        if len(answer) >= 30 and self.calculate_korean_ratio(answer) >= 0.8:
            return True
        
        return False
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화 - LLM 생성 결과 최적화"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 객관식: 숫자 추출
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            return ""
        
        else:
            # 주관식: LLM 생성 결과 최적화
            answer = self.clean_korean_text_premium(answer)
            
            # 정리 후 내용이 없으면 기본 답변
            if not answer:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 너무 짧은 답변 처리
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 길이 제한
            if len(answer) > 350:
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:3])
                if len(answer) > 350:
                    answer = answer[:350]
                    # 한글로 끝나도록 조정
                    last_korean = answer.rfind('다')
                    if last_korean > len(answer) - 10:
                        answer = answer[:last_korean + 1]
            
            # 문장 끝 처리
            if answer and not answer.endswith(('.', '다', '요', '함', '니다', '습니다')):
                answer += "."
            
            return answer
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        
        stats = {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats.get("question_type_accuracy", {"correct": 0, "total": 0}),
            "llm_generation_enhanced_rate": (self.processing_stats.get("llm_generation_enhanced", 0) / total) * 100,
            "text_post_processing_rate": (self.processing_stats.get("text_post_processing", 0) / total) * 100
        }
        
        return stats
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()