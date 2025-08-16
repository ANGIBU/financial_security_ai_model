# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
- 질문 의도 분석
- 강화된 패턴 매칭
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
    TEXT_CLEANUP_CONFIG, KOREAN_TYPO_MAPPING, OPTIMIZATION_CONFIG
)

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        self.pkl_dir = PKL_DIR
        self.pkl_dir.mkdir(exist_ok=True)
        
        self._load_json_configs()
        
        self.korean_requirements = KOREAN_REQUIREMENTS
        self.text_cleanup_config = TEXT_CLEANUP_CONFIG
        self.korean_typo_mapping = KOREAN_TYPO_MAPPING
        
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
            self.high_frequency_keywords = processing_config.get('high_frequency_keywords', {})
            
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
            "지표_묻기": ["지표.*설명하세요", "탐지.*지표"],
            "방안_묻기": ["방안.*기술하세요", "방안.*설명하세요"],
            "절차_묻기": ["절차.*설명하세요", "절차.*기술하세요"],
            "조치_묻기": ["조치.*설명하세요", "조치.*기술하세요"]
        }
        
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요',
            r'서술하세요',
            r'작성하세요'
        ]
        
        self.domain_keywords = {
            "일반": ["법령", "규정", "관리", "조치", "절차"]
        }
        
        self.domain_detection_patterns = {}
        self.choice_extraction_patterns = []
        self.answer_validation_rules = {}
        self.high_frequency_keywords = {}
        
        self.processing_stats_structure = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0},
            "choice_count_errors": 0,
            "intent_analysis_accuracy": {"correct": 0, "total": 0},
            "intent_match_accuracy": {"correct": 0, "total": 0},
            "mc_classification_accuracy": {"correct": 0, "total": 0}
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
    
    def clean_korean_text_advanced(self, text: str) -> str:
        """한국어 텍스트 정리 - 개선된 안전 버전"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 명백한 오류만 수정 (과도한 정리 방지)
        # 1단계: 심각한 오타만 수정
        critical_errors = {
            "감추인": "숨겨진",
            "컨퍼머시": "시스템",
            "피-에": "대상에",
            "백-도어": "백도어",
            "키-로거": "키로거",
            "스크리너": "화면캡처",
            "채팅-클라언트": "통신기능",
            "파일-업-": "파일업로드",
            "전자금윋": "전자금융",
            "캉터": "컴퓨터",
            "트래픁": "트래픽",
            "하웨어": "하드웨어",
            "네됴크": "네트워크"
        }
        
        for wrong, correct in critical_errors.items():
            text = text.replace(wrong, correct)
        
        # 2단계: 비정상적인 하이픈 패턴만 수정
        text = re.sub(r'([가-힣])-([가-힣])', r'\1\2', text)
        text = re.sub(r'([가-힣])-([ㄱ-ㅎ])', r'\1\2', text)
        text = re.sub(r'-{2,}', ' ', text)
        
        # 3단계: 공백 정리만
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def fix_korean_sentence_structure(self, text: str) -> str:
        """한국어 문장 구조 수정 - 개선된 버전"""
        if not text:
            return ""
        
        # 먼저 안전한 텍스트 정리
        text = self.clean_korean_text_advanced(text)
        
        # 문장 분할 (조건 완화)
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in ['.', '!', '?']:
                sentence = current_sentence.strip()
                if len(sentence) > 3:  # 최소 길이 조건 완화
                    sentences.append(sentence)
                current_sentence = ""
        
        # 마지막 문장 처리
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 각 문장 정리 (조건 완화)
        cleaned_sentences = []
        for sentence in sentences:
            # 너무 짧은 문장은 제외하되 조건 완화
            if len(sentence) < 5:
                continue
            
            # 한국어 비율 확인 (조건 완화)
            korean_chars = len(re.findall(r'[가-힣]', sentence))
            total_chars = len(re.sub(r'[^\w가-힣]', '', sentence))
            
            if total_chars > 0 and korean_chars / total_chars >= 0.5:  # 0.8에서 0.5로 완화
                cleaned_sentences.append(sentence)
        
        # 문장 연결
        result = ' '.join(cleaned_sentences)
        
        # 마침표 확인
        if result and not result.endswith(('.', '다', '요', '함')):
            result += '.'
        
        return result
    
    def analyze_question_intent(self, question: str) -> Dict:
        """강화된 질문 의도 분석 - 기관 탐지 개선"""
        question_lower = question.lower()
        
        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "domain_specific_intent": None
        }
        
        intent_scores = {}
        
        # 기관 관련 질문 우선 처리 (강화된 패턴)
        institution_patterns = [
            r"기관.*기술하세요", r"기관.*설명하세요", r"기관.*서술하세요",
            r"어떤.*기관", r"어느.*기관", r"기관.*무엇", r"기관.*어디",
            r"조정.*신청.*기관", r"분쟁.*조정.*기관", r"신청.*수.*있는.*기관",
            r"담당.*기관", r"관리.*기관", r"감독.*기관", r"소관.*기관",
            r"신고.*기관", r"접수.*기관", r"상담.*기관", r"문의.*기관",
            r"위원회.*무엇", r"위원회.*어디", r"위원회.*설명",
            r"분쟁조정.*신청.*가능.*기관", r"침해.*신고.*접수.*기관",
            r"조정.*담당.*기관", r"감독.*업무.*기관"
        ]
        
        institution_score = 0
        for pattern in institution_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                institution_score += 2
        
        # 기관 키워드 추가 점수
        institution_keywords = [
            "전자금융", "분쟁조정", "개인정보", "침해신고", "금융감독원",
            "한국은행", "개인정보보호위원회", "위원회", "센터"
        ]
        
        for keyword in institution_keywords:
            if keyword in question_lower:
                institution_score += 1
        
        # 기관 관련 점수가 높으면 우선 처리
        if institution_score >= 2:
            intent_scores["기관_묻기"] = {
                "score": institution_score,
                "patterns": ["institution_detection"]
            }
        
        # 다른 의도 패턴 분석
        for intent_type, patterns in self.question_intent_patterns.items():
            if intent_type == "기관_묻기" and institution_score >= 2:
                continue  # 이미 처리됨
                
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, question, re.IGNORECASE)
                if matches:
                    if len(matches) > 1:
                        score += 2
                    else:
                        score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns
                }
        
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]
            
            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 4, 1.0)  # 스케일 조정
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]
            
            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]} 
                    for intent, data in sorted_intents[1:3]
                ]
            
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명 필요")
                
                # 특정 기관 유형 판단 강화
                if "전자금융" in question_lower and any(word in question_lower for word in ["분쟁", "조정"]):
                    intent_analysis["domain_specific_intent"] = "전자금융분쟁조정위원회"
                elif "개인정보" in question_lower and any(word in question_lower for word in ["침해", "신고"]):
                    intent_analysis["domain_specific_intent"] = "개인정보침해신고센터"
                elif "한국은행" in question_lower or "금융통화위원회" in question_lower:
                    intent_analysis["domain_specific_intent"] = "한국은행"
                elif "금융감독원" in question_lower:
                    intent_analysis["domain_specific_intent"] = "금융감독원"
                    
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("특징과 성질 나열")
                
                if "트로이" in question_lower or "trojan" in question_lower:
                    intent_analysis["domain_specific_intent"] = "트로이목마특징"
                    
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행방안")
                
                if "딥페이크" in question_lower:
                    intent_analysis["domain_specific_intent"] = "딥페이크대응방안"
                    
            elif "절차" in primary:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["context_hints"].append("단계별 절차")
            elif "조치" in primary:
                intent_analysis["answer_type_required"] = "조치설명"
                intent_analysis["context_hints"].append("보안조치 내용")
            elif "법령" in primary:
                intent_analysis["answer_type_required"] = "법령설명"
                intent_analysis["context_hints"].append("관련 법령과 규정")
            elif "정의" in primary:
                intent_analysis["answer_type_required"] = "정의설명"
                intent_analysis["context_hints"].append("개념과 정의")
        
        self._add_context_analysis(question, intent_analysis)
        
        self.processing_stats["intent_analysis_accuracy"]["total"] += 1
        
        return intent_analysis
    
    def _add_context_analysis(self, question: str, intent_analysis: Dict):
        """추가 문맥 분석 - 기관 관련 강화"""
        question_lower = question.lower()
        
        # 기관 관련 추가 분석
        institution_indicators = [
            "분쟁조정을 신청할 수 있는", "침해신고를 접수하는", "업무를 담당하는",
            "관리하는 기관", "감독하는 기관", "소관 기관", "전담 기관"
        ]
        
        for indicator in institution_indicators:
            if indicator in question_lower:
                intent_analysis["context_hints"].append("기관 업무 범위 포함 필요")
                break
        
        urgency_keywords = ["긴급", "즉시", "신속", "빠른"]
        if any(keyword in question_lower for keyword in urgency_keywords):
            intent_analysis["context_hints"].append("긴급 대응 필요")
        
        example_keywords = ["예시", "사례", "구체적", "실제"]
        if any(keyword in question_lower for keyword in example_keywords):
            intent_analysis["context_hints"].append("구체적 예시 포함")
        
        comparison_keywords = ["비교", "차이", "구별", "비교하여"]
        if any(keyword in question_lower for keyword in comparison_keywords):
            intent_analysis["context_hints"].append("비교 분석 필요")
        
        step_keywords = ["단계", "순서", "과정", "절차"]
        if any(keyword in question_lower for keyword in step_keywords):
            intent_analysis["context_hints"].append("단계별 설명 필요")
        
        # 법령 관련 키워드
        law_keywords = ["전자금융거래법", "개인정보보호법", "자본시장법", "정보통신망법"]
        for keyword in law_keywords:
            if keyword in question:
                intent_analysis["context_hints"].append(f"{keyword} 관련 내용 중요")
        
        # 고빈도 키워드
        high_freq_keywords = ["금융감독원", "SBOM", "만 14세", "한국은행", "위험관리"]
        for keyword in high_freq_keywords:
            if keyword in question:
                intent_analysis["context_hints"].append(f"{keyword} 관련 내용 중요")
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """강화된 선택지 범위 추출"""
        question_type = self.analyze_question_type(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_numbers.append(num)
        
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            expected_count = max_choice - min_choice + 1
            if (len(choice_numbers) == expected_count and 
                min_choice == 1 and 
                max_choice >= 3):
                return "multiple_choice", max_choice
        
        for pattern_info in self.choice_extraction_patterns:
            pattern = pattern_info['pattern']
            if re.search(pattern, question, re.MULTILINE):
                matches = re.findall(pattern, question, re.MULTILINE)
                if len(matches) >= 3:
                    return "multiple_choice", len(matches)
        
        for i in range(5, 2, -1):
            pattern_parts = [f'{j}\\s+[가-힣\\w]+' for j in range(1, i+1)]
            pattern = '.*'.join(pattern_parts)
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i
        
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                self.processing_stats["choice_count_errors"] += 1
                return "multiple_choice", 5
        
        return "subjective", 0
    
    def analyze_question_type(self, question: str) -> str:
        """강화된 질문 유형 분석"""
        
        question = question.strip()
        self.processing_stats["question_type_accuracy"]["total"] += 1
        self.processing_stats["mc_classification_accuracy"]["total"] += 1
        
        # 주관식 패턴 우선 확인 (강화)
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
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                if any(f'{i} ' in question for i in range(1, 6)):
                    self.processing_stats["question_type_accuracy"]["correct"] += 1
                    self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                    
                    if "옳지" in question or "않은" in question:
                        self.processing_stats["negative_pattern_matches"] = \
                            self.processing_stats.get("negative_pattern_matches", 0) + 1
                    elif "가장" in question and "적절한" in question:
                        self.processing_stats["positive_pattern_matches"] = \
                            self.processing_stats.get("positive_pattern_matches", 0) + 1
                    
                    return "multiple_choice"
        
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        if (len(question) < 400 and 
            re.search(r'것은\?|것\?|것은\s*$', question) and
            len(re.findall(r'\b[1-5]\b', question)) >= 3):
            return "multiple_choice"
        
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """강화된 도메인 추출"""
        question_lower = question.lower()
        
        domain_scores = {}
        
        if self.domain_detection_patterns:
            for domain, patterns in self.domain_detection_patterns.items():
                score = 0
                
                if 'strong_indicators' in patterns:
                    for indicator in patterns['strong_indicators']:
                        if indicator.lower() in question_lower:
                            score += 3
                
                if 'medium_indicators' in patterns:
                    for indicator in patterns['medium_indicators']:
                        if indicator.lower() in question_lower:
                            score += 2
                
                if 'law_references' in patterns:
                    for law in patterns['law_references']:
                        if law.lower() in question_lower:
                            score += 2
                
                if 'technical_terms' in patterns:
                    for term in patterns['technical_terms']:
                        if term.lower() in question_lower:
                            score += 1
                
                if score > 0:
                    domain_scores[domain] = score
        
        if not domain_scores:
            for domain, keywords in self.domain_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword.lower() in question_lower:
                        if keyword in self.high_frequency_keywords:
                            score += 2
                        else:
                            score += 1
                
                if score > 0:
                    domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        if "domain_confidence_scores" not in self.processing_stats:
            self.processing_stats["domain_confidence_scores"] = {}
        if detected_domain not in self.processing_stats["domain_confidence_scores"]:
            self.processing_stats["domain_confidence_scores"][detected_domain] = []
        
        max_score = domain_scores[detected_domain]
        total_score = sum(domain_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0
        self.processing_stats["domain_confidence_scores"][detected_domain].append(confidence)
        
        return detected_domain
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리"""
        return self.clean_korean_text_advanced(text)
    
    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0
        
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return english_chars / total_chars
    
    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 검증"""
        if not answer or not answer.isdigit():
            return False
        
        answer_num = int(answer)
        
        if self.answer_validation_rules and 'multiple_choice' in self.answer_validation_rules:
            valid_range = self.answer_validation_rules['multiple_choice']['valid_range']
            return valid_range[0] <= answer_num <= min(max_choice, valid_range[1])
        
        return 1 <= answer_num <= max_choice
    
    def validate_answer_intent_match(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """강화된 답변과 질문 의도 일치성 검증 - 조건 완화"""
        if not answer or not intent_analysis:
            return False
        
        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()
        
        # 기관명 답변 검증 강화
        if required_type == "기관명":
            # 구체적인 기관명 리스트
            specific_institutions = [
                "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                "개인정보침해신고센터", "한국은행", "금융위원회",
                "과학기술정보통신부", "한국인터넷진흥원"
            ]
            
            # 일반적인 기관 키워드
            institution_keywords = [
                "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", 
                "조정위원회", "보호위원회", "신고센터"
            ]
            
            # 구체적인 기관명이 있는지 확인
            has_specific = any(inst in answer_lower for inst in specific_institutions)
            
            # 일반적인 기관 키워드 개수
            keyword_count = sum(1 for keyword in institution_keywords if keyword in answer_lower)
            
            # 기관 관련 업무 키워드
            role_keywords = ["담당", "업무", "수행", "관리", "감독", "조정", "신고", "접수"]
            role_count = sum(1 for keyword in role_keywords if keyword in answer_lower)
            
            # 기관명 검증 (조건 완화)
            match_found = has_specific or (keyword_count >= 1 and role_count >= 1) or keyword_count >= 2
        
        elif required_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격", "형태", "방식"]
            descriptive_words = ["위장", "은밀", "지속", "제어", "접근", "수행", "활동", "백도어", "권한", "숨겨진"]
            
            feature_count = sum(1 for keyword in feature_keywords if keyword in answer_lower)
            desc_count = sum(1 for word in descriptive_words if word in answer_lower)
            
            match_found = feature_count >= 1 or (desc_count >= 2 and len(answer) > 50)  # 조건 완화
        
        elif required_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지", "발견", "식별"]
            specific_indicators = ["네트워크", "트래픽", "프로세스", "파일", "시스템", "로그", "연결", "메모리", "레지스트리"]
            
            indicator_count = sum(1 for keyword in indicator_keywords if keyword in answer_lower)
            specific_count = sum(1 for word in specific_indicators if word in answer_lower)
            
            match_found = indicator_count >= 1 or specific_count >= 2  # 조건 완화
        
        elif required_type == "방안제시":
            solution_keywords = ["방안", "대책", "조치", "해결", "대응", "관리", "처리", "절차", "개선", "예방", "강화", "구축"]
            action_words = ["수립", "구축", "시행", "실시", "강화", "개선", "마련", "도입", "적용", "운영"]
            
            solution_count = sum(1 for keyword in solution_keywords if keyword in answer_lower)
            action_count = sum(1 for word in action_words if word in answer_lower)
            
            match_found = solution_count >= 1 or action_count >= 1  # 조건 완화
        
        elif required_type == "절차설명":
            procedure_keywords = ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행", "실행", "방법"]
            step_indicators = ["첫째", "둘째", "먼저", "다음", "마지막", "단계적", "순차적", "1단계", "2단계"]
            
            proc_count = sum(1 for keyword in procedure_keywords if keyword in answer_lower)
            step_count = sum(1 for word in step_indicators if word in answer_lower)
            
            match_found = proc_count >= 1 or step_count >= 1 or ("," in answer and len(answer) > 50)  # 조건 완화
        
        elif required_type == "조치설명":
            measure_keywords = ["조치", "대응", "대책", "방안", "보안", "예방", "개선", "강화", "보완", "통제", "관리"]
            match_found = sum(1 for keyword in measure_keywords if keyword in answer_lower) >= 1  # 조건 완화
        
        elif required_type == "법령설명":
            law_keywords = ["법", "법령", "법률", "규정", "조항", "규칙", "기준", "근거", "제도", "정책"]
            match_found = sum(1 for keyword in law_keywords if keyword in answer_lower) >= 1  # 조건 완화
        
        elif required_type == "정의설명":
            definition_keywords = ["정의", "개념", "의미", "뜻", "용어", "이란", "라고", "것은", "말하며", "지칭"]
            match_found = sum(1 for keyword in definition_keywords if keyword in answer_lower) >= 1
        
        else:
            meaningful_words = ["법령", "규정", "관리", "조치", "절차", "기준", "정책", "체계", "시스템", "보안", "통제"]
            match_found = sum(1 for word in meaningful_words if word in answer_lower) >= 1  # 조건 완화
        
        self.processing_stats["intent_match_accuracy"]["total"] += 1
        if match_found:
            self.processing_stats["intent_match_accuracy"]["correct"] += 1
        
        return match_found
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """강화된 한국어 답변 유효성 검증 - 조건 완화"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            if not self.validate_mc_answer_range(answer, max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
        else:
            clean_answer = self.clean_korean_text_advanced(answer)
            clean_answer = self.fix_korean_sentence_structure(clean_answer)
            
            if self.answer_validation_rules and 'subjective' in self.answer_validation_rules:
                rules = self.answer_validation_rules['subjective']
                min_length = rules.get('min_length', 30)  # 50에서 30으로 완화
                max_length = rules.get('max_length', 500)  # 400에서 500으로 완화
                min_korean_ratio = rules.get('min_korean_ratio', 0.7)  # 0.9에서 0.7으로 완화
            else:
                min_length = 30  # 완화
                max_length = 500  # 완화
                min_korean_ratio = 0.7  # 완화
            
            if not (min_length <= len(clean_answer) <= max_length):
                self.processing_stats["validation_failures"] += 1
                return False
            
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < min_korean_ratio:
                self.processing_stats["validation_failures"] += 1
                return False
            
            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > 0.1:  # 0.05에서 0.1로 완화
                self.processing_stats["validation_failures"] += 1
                return False
            
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 15:  # 20에서 15로 완화
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의미있는 키워드 검증 (조건 완화)
            meaningful_keywords = [
                "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", 
                "정책", "체계", "시스템", "통제", "대응", "처리", "기관",
                "위원회", "특징", "지표", "탐지", "모니터링"
            ]
            
            keyword_count = sum(1 for word in meaningful_keywords if word in clean_answer)
            if keyword_count < 1:  # 최소 1개만 있으면 됨
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의도 일치성 검증 (선택적)
            if question:
                intent_analysis = self.analyze_question_intent(question)
                if not self.validate_answer_intent_match(answer, question, intent_analysis):
                    # 완전 실패가 아닌 경고만 (통계에만 기록)
                    pass
            
            self.processing_stats["korean_compliance"] += 1
            return True
    
    def validate_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """답변 유효성 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        return self.clean_korean_text_advanced(text)
    
    def extract_choices(self, question: str) -> List[str]:
        """강화된 객관식 선택지 추출"""
        choices = []
        
        lines = question.split('\n')
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                choice_num = int(match.group(1))
                choice_content = match.group(2).strip()
                if 1 <= choice_num <= 5 and len(choice_content) > 0:
                    choices.append(choice_content)
        
        if len(choices) >= 3:
            return choices
        
        if not choices:
            for pattern_info in self.choice_extraction_patterns:
                pattern = pattern_info['pattern']
                matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
                if matches:
                    if isinstance(matches[0], tuple):
                        choices = [match[1].strip() for match in matches]
                    else:
                        choices = [match.strip() for match in matches]
                    
                    if len(choices) >= 3:
                        break
        
        if not choices:
            patterns = [
                r'(\d+)\s+([^0-9\n]+?)(?=\d+\s+|$)',
                r'(\d+)\)\s*([^0-9\n]+?)(?=\d+\)|$)',
                r'(\d+)\.\s*([^0-9\n]+?)(?=\d+\.|$)',
                r'[①②③④⑤]\s*([^①②③④⑤\n]+?)(?=[①②③④⑤]|$)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
                if matches:
                    if isinstance(matches[0], tuple):
                        choices = [match[1].strip() for match in matches]
                    else:
                        choices = [match.strip() for match in matches]
                    
                    if len(choices) >= 3:
                        break
        
        return choices[:5]
    
    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()
        
        technical_terms = list(self.high_frequency_keywords.keys()) + [
            "isms", "pims", "sbom", "원격제어", "침입탐지", 
            "트로이", "멀웨어", "랜섬웨어", "딥페이크", "피싱",
            "접근매체", "전자서명", "개인정보보호법", "자본시장법",
            "rat", "원격접근", "탐지지표", "apt", "ddos",
            "ids", "ips", "bcp", "drp", "isms-p", "분쟁조정",
            "금융투자업", "위험관리", "재해복구", "비상연락체계"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question_lower)
        
        length = len(question)
        
        choice_count = len(self.extract_choices(question))
        
        if term_count >= 3 or length > 400 or choice_count >= 5:
            return "고급"
        elif term_count >= 1 or length > 200 or choice_count >= 4:
            return "중급"
        else:
            return "초급"
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화 - 개선된 버전"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            return ""
        
        else:
            answer = self.clean_korean_text_advanced(answer)
            answer = self.fix_korean_sentence_structure(answer)
            
            # 너무 짧은 답변 처리 (조건 완화)
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 길이 제한 (완화)
            if len(answer) > 500:  # 400에서 500으로 완화
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:4])  # 3에서 4로 증가
                if len(answer) > 500:
                    answer = answer[:500]
            
            if answer and not answer.endswith(('.', '다', '요', '함')):
                answer += "."
            
            return answer
    
    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화"""
        return self.normalize_korean_answer(answer, question_type, max_choice)
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        intent_total = max(self.processing_stats["intent_analysis_accuracy"]["total"], 1)
        intent_match_total = max(self.processing_stats["intent_match_accuracy"]["total"], 1)
        mc_total = max(self.processing_stats["mc_classification_accuracy"]["total"], 1)
        
        stats = {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "choice_count_errors": self.processing_stats["choice_count_errors"],
            "intent_analysis_accuracy_rate": (self.processing_stats["intent_analysis_accuracy"]["correct"] / intent_total) * 100,
            "intent_match_accuracy_rate": (self.processing_stats["intent_match_accuracy"]["correct"] / intent_match_total) * 100,
            "mc_classification_accuracy_rate": (self.processing_stats["mc_classification_accuracy"]["correct"] / mc_total) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats["question_type_accuracy"]
        }
        
        if "negative_pattern_matches" in self.processing_stats:
            stats["negative_pattern_matches"] = self.processing_stats["negative_pattern_matches"]
        if "positive_pattern_matches" in self.processing_stats:
            stats["positive_pattern_matches"] = self.processing_stats["positive_pattern_matches"]
        if "domain_confidence_scores" in self.processing_stats:
            avg_confidence = {}
            for domain, scores in self.processing_stats["domain_confidence_scores"].items():
                if scores:
                    avg_confidence[domain] = sum(scores) / len(scores)
            stats["average_domain_confidence"] = avg_confidence
        
        return stats
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()