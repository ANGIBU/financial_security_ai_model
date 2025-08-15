# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
- 질문 의도 분석 강화 (정확도 개선)
"""

import re
import pickle
import os
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 객관식 패턴 (개선된 버전)
        self.mc_patterns = [
            r'①.*②.*③.*④.*⑤',  # 동그라미 숫자
            r'1\s+[가-힣].*2\s+[가-힣].*3\s+[가-힣].*4\s+[가-힣].*5\s+[가-힣]',  # 번호 + 한글
            r'1\s+.*2\s+.*3\s+.*4\s+.*5\s+',  # 번호 공백 형식
            r'1\.\s*.*2\.\s*.*3\.\s*.*4\.\s*.*5\.',  # 1. 2. 3. 형식
            r'1\)\s*.*2\)\s*.*3\)\s*.*4\)\s*.*5\)',  # 1) 2) 3) 형식
        ]
        
        # 객관식 키워드 패턴
        self.mc_keywords = [
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것', 
            r'옳지.*않는.*것',
            r'틀린.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것',
            r'올바른.*것',
            r'가장.*적절한.*것',
            r'가장.*옳은.*것',
            r'구분.*해당하지.*않는.*것',
            r'다음.*중.*것은',
            r'다음.*중.*것',
            r'다음.*보기.*중'
        ]
        
        # 질문 의도 분석 패턴 (대폭 강화)
        self.question_intent_patterns = {
            "기관_묻기": [
                # 직접적인 기관 질문
                r'기관.*기술하세요',
                r'기관.*설명하세요',
                r'기관.*서술하세요',
                r'기관.*무엇',
                r'어떤.*기관',
                r'어느.*기관',
                r'기관.*어디',
                
                # 조정/분쟁 관련
                r'조정.*신청.*기관',
                r'분쟁.*조정.*기관',
                r'신청.*수.*있는.*기관',
                r'분쟁.*해결.*기관',
                r'조정.*담당.*기관',
                
                # 감독/관리 기관
                r'감독.*기관',
                r'관리.*기관',
                r'담당.*기관',
                r'주관.*기관',
                r'소관.*기관',
                
                # 신고/접수 기관
                r'신고.*기관',
                r'접수.*기관',
                r'상담.*기관',
                r'문의.*기관',
                
                # 위원회 관련
                r'위원회.*무엇',
                r'위원회.*어디',
                r'위원회.*설명',
                
                # 전자금융 관련 특화
                r'전자금융.*분쟁.*기관',
                r'전자금융.*조정.*기관',
                
                # 개인정보 관련 특화
                r'개인정보.*신고.*기관',
                r'개인정보.*보호.*기관',
                r'개인정보.*침해.*기관',
                
                # 추가 패턴 (강화)
                r'설치.*기관',
                r'운영.*기관',
                r'지정.*기관',
                r'관할.*기관',
                r'소속.*기관'
            ],
            "특징_묻기": [
                r'특징.*설명하세요',
                r'특징.*기술하세요',
                r'특징.*서술하세요',
                r'어떤.*특징',
                r'주요.*특징',
                r'특징.*무엇',
                r'성격.*설명',
                r'성질.*설명',
                r'속성.*설명',
                r'특성.*설명',
                r'특성.*무엇',
                r'성격.*무엇',
                r'특성.*기술',
                r'속성.*기술',
                r'고유.*특성',
                r'독특.*특징',
                r'핵심.*특징',
                r'본질.*특성',
                r'기본.*특징',
                r'고유.*속성'
            ],
            "지표_묻기": [
                r'지표.*설명하세요',
                r'탐지.*지표',
                r'주요.*지표',
                r'어떤.*지표',
                r'지표.*무엇',
                r'징후.*설명',
                r'신호.*설명',
                r'패턴.*설명',
                r'행동.*패턴',
                r'활동.*패턴',
                r'모니터링.*지표',
                r'관찰.*지표',
                r'식별.*지표',
                r'발견.*방법',
                r'탐지.*방법',
                r'확인.*방법',
                r'판단.*지표',
                r'추적.*지표',
                r'감시.*지표',
                r'체크.*지표'
            ],
            "방안_묻기": [
                r'방안.*기술하세요',
                r'방안.*설명하세요',
                r'대응.*방안',
                r'해결.*방안',
                r'관리.*방안',
                r'어떤.*방안',
                r'대책.*설명',
                r'조치.*방안',
                r'처리.*방안',
                r'개선.*방안',
                r'예방.*방안',
                r'보완.*방안',
                r'강화.*방안',
                r'구체적.*방안',
                r'실행.*방안',
                r'운영.*방안',
                r'시행.*방안',
                r'추진.*방안',
                r'도입.*방안',
                r'적용.*방안'
            ],
            "절차_묻기": [
                r'절차.*설명하세요',
                r'절차.*기술하세요',
                r'어떤.*절차',
                r'처리.*절차',
                r'진행.*절차',
                r'수행.*절차',
                r'실행.*절차',
                r'과정.*설명',
                r'단계.*설명',
                r'프로세스.*설명',
                r'순서.*설명',
                r'절차.*무엇',
                r'단계별.*절차',
                r'체계적.*절차',
                r'표준.*절차',
                r'운영.*절차',
                r'업무.*절차',
                r'처리.*과정',
                r'수행.*과정'
            ],
            "조치_묻기": [
                r'조치.*설명하세요',
                r'조치.*기술하세요',
                r'어떤.*조치',
                r'보안.*조치',
                r'대응.*조치',
                r'예방.*조치',
                r'개선.*조치',
                r'강화.*조치',
                r'보완.*조치',
                r'필요.*조치',
                r'적절.*조치',
                r'즉시.*조치',
                r'사전.*조치',
                r'사후.*조치',
                r'긴급.*조치',
                r'차단.*조치',
                r'방어.*조치',
                r'보호.*조치',
                r'통제.*조치'
            ],
            "법령_묻기": [
                r'법령.*설명',
                r'법률.*설명',
                r'규정.*설명',
                r'조항.*설명',
                r'규칙.*설명',
                r'기준.*설명',
                r'법적.*근거',
                r'관련.*법',
                r'적용.*법',
                r'준거.*법',
                r'근거.*법령',
                r'법률.*근거',
                r'규정.*근거',
                r'조항.*근거',
                r'법령.*조항',
                r'법률.*조항',
                r'규정.*조항'
            ],
            "정의_묻기": [
                r'정의.*설명',
                r'개념.*설명',
                r'의미.*설명',
                r'뜻.*설명',
                r'무엇.*의미',
                r'무엇.*뜻',
                r'용어.*설명',
                r'개념.*무엇',
                r'정의.*무엇',
                r'의미.*무엇',
                r'뜻.*무엇',
                r'설명.*개념',
                r'설명.*정의',
                r'설명.*의미',
                r'해석.*의미',
                r'이해.*개념'
            ]
        }
        
        # 주관식 패턴 (확장)
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요', 
            r'서술하세요',
            r'작성하세요',
            r'무엇인가요',
            r'어떻게.*해야.*하며',
            r'방안을.*기술',
            r'대응.*방안',
            r'특징.*다음과.*같',
            r'탐지.*지표',
            r'행동.*패턴',
            r'분석하여.*제시',
            r'조치.*사항',
            r'제시하시오',
            r'논하시오',
            r'답하시오'
        ]
        
        # 도메인 키워드 (확장)
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "동의", "법정대리인", "아동",
                "개인정보처리자", "열람권", "정정삭제권", "처리정지권",
                "개인정보보호위원회", "손해배상", "처리방침",
                "개인정보영향평가", "개인정보보호책임자", "개인정보침해신고센터"
            ],
            "전자금융": [
                "전자금융", "전자서명", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서", "분쟁조정",
                "전자지급수단", "전자화폐", "금융감독원", "한국은행",
                "전자금융업", "전자금융분쟁조정위원회", "전자금융거래",
                "전자금융업무", "전자금융서비스"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱",
                "스미싱", "랜섬웨어", "해킹", "딥페이크", "원격제어",
                "RAT", "원격접근", "봇넷", "백도어", "루트킷",
                "취약점", "제로데이", "사회공학", "APT", "DDoS",
                "침입탐지", "침입방지", "보안관제"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지",
                "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP",
                "정보보안관리체계"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "소비자금융업", "보험중개업", "자본시장법",
                "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용",
                "리스크", "내부통제", "컴플라이언스", "위험식별",
                "위험분석", "위험모니터링", "위험회피", "위험전가",
                "위험감소", "잔여위험", "위험성향"
            ]
        }
        
        # 한국어 전용 검증 기준
        self.korean_requirements = {
            "min_korean_ratio": 0.8,  # 최소 한국어 비율 80%
            "max_english_ratio": 0.1,  # 최대 영어 비율 10%
            "min_length": 30,  # 최소 길이
            "max_length": 500  # 최대 길이
        }
        
        # 처리 통계
        self.processing_stats = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0},
            "choice_count_errors": 0,
            "intent_analysis_accuracy": {"correct": 0, "total": 0},
            "intent_match_accuracy": {"correct": 0, "total": 0},  # 의도 일치 정확도 추가
            "pattern_matching_score": 0,  # 패턴 매칭 점수
            "semantic_analysis_score": 0,  # 의미 분석 점수
            "context_understanding_score": 0  # 컨텍스트 이해 점수
        }
        
        # 이전 처리 기록 로드
        self._load_processing_history()
    
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
    
    def analyze_question_intent(self, question: str) -> Dict:
        """질문 의도 분석 (강화)"""
        question_lower = question.lower()
        
        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],  # 부차적 의도들
            "context_hints": [],  # 문맥 힌트
            "pattern_strength": {},  # 패턴별 강도
            "semantic_markers": [],  # 의미적 마커
            "domain_context": "일반"  # 도메인 컨텍스트
        }
        
        # 각 의도 패턴별 점수 계산 (개선)
        intent_scores = {}
        pattern_strengths = {}
        
        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []
            pattern_weights = []
            
            for pattern in patterns:
                matches = re.findall(pattern, question, re.IGNORECASE)
                if matches:
                    # 패턴 매칭 강도 계산 (신규)
                    pattern_length = len(pattern)
                    match_count = len(matches)
                    pattern_weight = pattern_length * 0.1 + match_count * 0.5
                    
                    if match_count > 1:
                        score += pattern_weight * 2  # 여러 번 매칭되면 가중치
                    else:
                        score += pattern_weight
                    
                    matched_patterns.append(pattern)
                    pattern_weights.append(pattern_weight)
            
            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns
                }
                pattern_strengths[intent_type] = sum(pattern_weights) / len(pattern_weights) if pattern_weights else 0
        
        # 의미적 분석 추가 (신규)
        semantic_score = self._analyze_semantic_markers(question, intent_scores)
        intent_analysis["semantic_markers"] = semantic_score["markers"]
        
        # 컨텍스트 이해 분석 (신규)
        context_score = self._analyze_context_understanding(question, intent_scores)
        intent_analysis["context_hints"].extend(context_score["hints"])
        
        # 가장 높은 점수의 의도 선택 (강화)
        if intent_scores:
            # 의미적 점수와 패턴 점수 통합
            for intent_type in intent_scores:
                if intent_type in semantic_score["intent_boost"]:
                    intent_scores[intent_type]["score"] += semantic_score["intent_boost"][intent_type]
                if intent_type in context_score["intent_boost"]:
                    intent_scores[intent_type]["score"] += context_score["intent_boost"][intent_type]
            
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]
            
            intent_analysis["primary_intent"] = best_intent[0]
            # 신뢰도 계산 개선
            base_confidence = min(best_intent[1]["score"] / 5.0, 1.0)  # 기본 신뢰도
            semantic_boost = semantic_score.get("confidence_boost", 0.0)
            context_boost = context_score.get("confidence_boost", 0.0)
            intent_analysis["intent_confidence"] = min(base_confidence + semantic_boost + context_boost, 1.0)
            
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]
            intent_analysis["pattern_strength"] = pattern_strengths
            
            # 부차적 의도들도 기록
            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]} 
                    for intent, data in sorted_intents[1:3]  # 상위 2개
                ]
            
            # 답변 유형 결정 (더 세분화)
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명 필요")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("특징과 성질 나열")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행방안")
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
        
        # 도메인 컨텍스트 설정
        intent_analysis["domain_context"] = self._determine_domain_context(question)
        
        # 추가 문맥 분석
        self._add_enhanced_context_analysis(question, intent_analysis)
        
        # 통계 업데이트
        self.processing_stats["intent_analysis_accuracy"]["total"] += 1
        self.processing_stats["pattern_matching_score"] += len(intent_analysis["detected_patterns"])
        self.processing_stats["semantic_analysis_score"] += len(intent_analysis["semantic_markers"])
        self.processing_stats["context_understanding_score"] += len(intent_analysis["context_hints"])
        
        return intent_analysis
    
    def _analyze_semantic_markers(self, question: str, intent_scores: Dict) -> Dict:
        """의미적 마커 분석 (신규)"""
        semantic_analysis = {
            "markers": [],
            "intent_boost": {},
            "confidence_boost": 0.0
        }
        
        question_lower = question.lower()
        
        # 의미적 키워드 그룹
        semantic_groups = {
            "기관_의미": ["위원회", "기관", "부서", "조직", "담당", "관할", "소관"],
            "특징_의미": ["특성", "성질", "속성", "기능", "역할", "특색", "성격"],
            "지표_의미": ["신호", "징후", "표시", "증상", "단서", "흔적", "패턴"],
            "방안_의미": ["대책", "해법", "솔루션", "방법", "수단", "전략", "계획"],
            "절차_의미": ["과정", "단계", "순서", "프로세스", "워크플로", "흐름"],
            "조치_의미": ["대응", "행동", "실행", "시행", "적용", "운영", "관리"]
        }
        
        # 각 그룹별 매칭 점수 계산
        for group_name, keywords in semantic_groups.items():
            found_keywords = [kw for kw in keywords if kw in question_lower]
            if found_keywords:
                semantic_analysis["markers"].extend(found_keywords)
                
                # 의도 부스트 계산
                intent_base = group_name.split("_")[0]
                intent_key = f"{intent_base}_묻기"
                if intent_key in intent_scores:
                    boost_score = len(found_keywords) * 0.3
                    semantic_analysis["intent_boost"][intent_key] = boost_score
                    semantic_analysis["confidence_boost"] += boost_score * 0.1
        
        return semantic_analysis
    
    def _analyze_context_understanding(self, question: str, intent_scores: Dict) -> Dict:
        """컨텍스트 이해 분석 (신규)"""
        context_analysis = {
            "hints": [],
            "intent_boost": {},
            "confidence_boost": 0.0
        }
        
        question_lower = question.lower()
        
        # 컨텍스트 힌트 패턴
        context_patterns = {
            "구체성_요구": ["구체적", "상세히", "자세히", "세부적", "명확히"],
            "예시_요구": ["예시", "사례", "실제", "예를", "구체적"],
            "비교_요구": ["비교", "차이", "구별", "비교하여", "다른점"],
            "단계_요구": ["단계", "순서", "과정", "절차", "프로세스"],
            "긴급성_표시": ["긴급", "즉시", "신속", "빠른", "urgent"],
            "완전성_요구": ["모든", "전체", "완전한", "총괄적", "포괄적"]
        }
        
        for pattern_type, keywords in context_patterns.items():
            found = [kw for kw in keywords if kw in question_lower]
            if found:
                if pattern_type == "구체성_요구":
                    context_analysis["hints"].append("구체적 세부사항 필요")
                elif pattern_type == "예시_요구":
                    context_analysis["hints"].append("구체적 예시 포함")
                elif pattern_type == "비교_요구":
                    context_analysis["hints"].append("비교 분석 필요")
                elif pattern_type == "단계_요구":
                    context_analysis["hints"].append("단계별 설명 필요")
                elif pattern_type == "긴급성_표시":
                    context_analysis["hints"].append("긴급 대응 필요")
                elif pattern_type == "완전성_요구":
                    context_analysis["hints"].append("포괄적 답변 필요")
                
                # 컨텍스트 기반 의도 부스트
                if pattern_type == "단계_요구":
                    if "절차_묻기" in intent_scores:
                        context_analysis["intent_boost"]["절차_묻기"] = 0.5
                        context_analysis["confidence_boost"] += 0.1
        
        return context_analysis
    
    def _determine_domain_context(self, question: str) -> str:
        """도메인 컨텍스트 결정 (신규)"""
        question_lower = question.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "일반"
    
    def _add_enhanced_context_analysis(self, question: str, intent_analysis: Dict):
        """강화된 컨텍스트 분석 (신규)"""
        question_lower = question.lower()
        
        # 질문 복잡도 분석
        complexity_indicators = ["복합적", "종합적", "다각적", "전반적", "포괄적"]
        if any(indicator in question_lower for indicator in complexity_indicators):
            intent_analysis["context_hints"].append("복합적 분석 필요")
        
        # 시급성 분석
        urgency_keywords = ["긴급", "즉시", "신속", "빠른", "시급"]
        if any(keyword in question_lower for keyword in urgency_keywords):
            intent_analysis["context_hints"].append("긴급 대응 필요")
        
        # 범위 분석
        scope_keywords = ["전체", "모든", "모든", "전반", "총괄"]
        if any(keyword in question_lower for keyword in scope_keywords):
            intent_analysis["context_hints"].append("전체적 접근 필요")
        
        # 전문성 수준 분석
        expert_keywords = ["전문적", "고도의", "심화된", "세밀한", "정밀한"]
        if any(keyword in question_lower for keyword in expert_keywords):
            intent_analysis["context_hints"].append("고도 전문성 필요")
        
        # 실무 중심 분석
        practical_keywords = ["실무", "실제", "현실적", "적용", "실행"]
        if any(keyword in question_lower for keyword in practical_keywords):
            intent_analysis["context_hints"].append("실무적 접근 필요")
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출 (개선된 버전)"""
        question_type = self.analyze_question_type(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        # 줄별로 분석하여 선택지 번호 추출
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            # 선택지 패턴: 숫자 + 공백 + 내용
            match = re.match(r'^(\d+)\s+', line.strip())
            if match:
                choice_numbers.append(int(match.group(1)))
        
        # 연속된 선택지인지 확인
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            # 연속성 검증
            expected_count = max_choice - min_choice + 1
            if len(choice_numbers) == expected_count and min_choice == 1:
                return "multiple_choice", max_choice
        
        # 폴백: 전통적인 패턴으로 확인
        for i in range(5, 2, -1):  # 5개부터 3개까지 확인
            pattern = r'1\s.*' + '.*'.join([f'{j}\s' for j in range(2, i+1)])
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i
        
        # 객관식 키워드가 있지만 선택지를 찾을 수 없는 경우
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                self.processing_stats["choice_count_errors"] += 1
                return "multiple_choice", 5  # 기본값
        
        return "subjective", 0
    
    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석 (개선된 버전)"""
        
        question = question.strip()
        self.processing_stats["question_type_accuracy"]["total"] += 1
        
        # 1차: 명확한 선택지 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        # 2차: 줄별 선택지 분석
        lines = question.split('\n')
        choice_lines = 0
        for line in lines:
            if re.match(r'^\s*[1-5][\s\.]\s*', line):
                choice_lines += 1
        
        if choice_lines >= 3:  # 3개 이상의 선택지가 있으면 객관식
            self.processing_stats["question_type_accuracy"]["correct"] += 1
            return "multiple_choice"
        
        # 3차: 객관식 키워드 확인
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지가 있는지 추가 확인
                if any(str(i) in question for i in range(1, 6)):
                    self.processing_stats["question_type_accuracy"]["correct"] += 1
                    return "multiple_choice"
        
        # 4차: 주관식 패턴 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 5차: 질문 구조 분석
        # 선택지 번호만 단순 카운트
        number_count = len(re.findall(r'\b[1-5]\b', question))
        if number_count >= 3 and len(question) < 300:
            return "multiple_choice"
        
        # 6차: "것은?" "것?" 패턴과 길이로 추가 판단
        if re.search(r'것은\?|것\?|것은\s*$', question):
            if len(question) < 300 and any(str(i) in question for i in range(1, 6)):
                return "multiple_choice"
        
        # 기본값: 주관식
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """도메인 추출 (개선)"""
        question_lower = question.lower()
        
        # 각 도메인별 키워드 매칭 점수 계산
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드는 가중치 부여
                    if keyword in ["개인정보보호법", "전자금융거래법", "자본시장법", "ISMS"]:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        # 가장 높은 점수의 도메인 반환
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 통계 업데이트
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리 (강화)"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 깨진 문자 및 인코딩 오류 처리
        text = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', text)
        
        # 영어 문자 제거 (한국어 답변을 위해)
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # 중국어 제거
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # 특수 기호 제거
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)
        
        # 반복 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
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
        return 1 <= answer_num <= max_choice
    
    def validate_answer_intent_match(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """답변과 질문 의도 일치성 검증 (강화)"""
        if not answer or not intent_analysis:
            return False
        
        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()
        primary_intent = intent_analysis.get("primary_intent", "일반")
        confidence = intent_analysis.get("intent_confidence", 0.0)
        
        # 신뢰도가 낮으면 관대한 검증
        if confidence < 0.5:
            relaxed_validation = self._relaxed_intent_validation(answer, primary_intent)
            self._update_intent_match_stats(relaxed_validation)
            return relaxed_validation
        
        # 기관명이 필요한 경우 (강화)
        if required_type == "기관명":
            institution_keywords = [
                "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", 
                "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                "개인정보침해신고센터", "한국은행", "금융위원회", "과학기술정보통신부"
            ]
            match_found = any(keyword in answer_lower for keyword in institution_keywords)
            
            # 의도 일치 정확도 업데이트
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 특징 설명이 필요한 경우
        elif required_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격"]
            match_found = any(keyword in answer_lower for keyword in feature_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 지표 나열이 필요한 경우
        elif required_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지", "발견", "식별"]
            match_found = any(keyword in answer_lower for keyword in indicator_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 방안 제시가 필요한 경우
        elif required_type == "방안제시":
            solution_keywords = ["방안", "대책", "조치", "해결", "대응", "관리", "처리", "절차", "개선", "예방"]
            match_found = any(keyword in answer_lower for keyword in solution_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 절차 설명이 필요한 경우
        elif required_type == "절차설명":
            procedure_keywords = ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행", "실행"]
            match_found = any(keyword in answer_lower for keyword in procedure_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 조치 설명이 필요한 경우
        elif required_type == "조치설명":
            measure_keywords = ["조치", "대응", "대책", "방안", "보안", "예방", "개선", "강화", "보완"]
            match_found = any(keyword in answer_lower for keyword in measure_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 법령 설명이 필요한 경우
        elif required_type == "법령설명":
            law_keywords = ["법", "법령", "법률", "규정", "조항", "규칙", "기준", "근거"]
            match_found = any(keyword in answer_lower for keyword in law_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 정의 설명이 필요한 경우
        elif required_type == "정의설명":
            definition_keywords = ["정의", "개념", "의미", "뜻", "용어", "개념"]
            match_found = any(keyword in answer_lower for keyword in definition_keywords)
            self._update_intent_match_stats(match_found)
            return match_found
        
        # 기본적으로 통과
        self._update_intent_match_stats(True)
        return True
    
    def _relaxed_intent_validation(self, answer: str, primary_intent: str) -> bool:
        """관대한 의도 검증 (신뢰도 낮을 때)"""
        answer_lower = answer.lower()
        
        # 기본적인 전문용어 포함 여부만 확인
        general_keywords = ["법령", "규정", "조치", "관리", "절차", "기준", "정책", "체계"]
        basic_match = any(keyword in answer_lower for keyword in general_keywords)
        
        # 의도별 최소 요구사항
        if "기관" in primary_intent:
            return "기관" in answer_lower or "위원회" in answer_lower or basic_match
        elif "특징" in primary_intent:
            return "특징" in answer_lower or "특성" in answer_lower or basic_match
        elif "지표" in primary_intent:
            return "지표" in answer_lower or "탐지" in answer_lower or basic_match
        elif "방안" in primary_intent:
            return "방안" in answer_lower or "조치" in answer_lower or basic_match
        else:
            return basic_match
    
    def _update_intent_match_stats(self, is_match: bool):
        """의도 일치 통계 업데이트"""
        self.processing_stats["intent_match_accuracy"]["total"] += 1
        if is_match:
            self.processing_stats["intent_match_accuracy"]["correct"] += 1
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """한국어 답변 유효성 검증 (강화)"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            # 객관식: 지정된 범위의 숫자
            if not self.validate_mc_answer_range(answer, max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
        else:
            # 주관식: 한국어 전용 검증 + 의도 일치성 검증
            clean_answer = self.clean_korean_text(answer)
            
            # 길이 검증
            if not (self.korean_requirements["min_length"] <= len(clean_answer) <= self.korean_requirements["max_length"]):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 한국어 비율 검증
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < self.korean_requirements["min_korean_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 영어 비율 검증
            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > self.korean_requirements["max_english_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 최소 한국어 문자 수 검증
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 20:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의미 있는 내용인지 확인
            meaningful_keywords = ["법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", "정책", "체계", "시스템", "통제"]
            if not any(word in clean_answer for word in meaningful_keywords):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 질문 의도 일치성 검증 (강화)
            if question:
                intent_analysis = self.analyze_question_intent(question)
                if not self.validate_answer_intent_match(answer, question, intent_analysis):
                    self.processing_stats["validation_failures"] += 1
                    return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
    
    def validate_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """답변 유효성 검증 (한국어 전용)"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리 (한국어 전용)"""
        return self.clean_korean_text(text)
    
    def extract_choices(self, question: str) -> List[str]:
        """객관식 선택지 추출"""
        choices = []
        
        # 줄별로 선택지 추출
        lines = question.split('\n')
        for line in lines:
            match = re.match(r'^(\d+)\s+(.+)', line.strip())
            if match:
                choice_num = int(match.group(1))
                choice_content = match.group(2).strip()
                if choice_num <= 5:  # 5번까지만
                    choices.append(choice_content)
        
        # 전통적인 패턴으로도 확인
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
                    
                    # 5개 선택지가 모두 있는지 확인
                    if len(choices) >= 3:
                        break
        
        return choices[:5]  # 최대 5개 선택지
    
    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()
        
        # 전문 용어 개수
        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", 
            "트로이", "멀웨어", "랜섬웨어", "딥페이크", "피싱",
            "접근매체", "전자서명", "개인정보보호법", "자본시장법",
            "rat", "원격접근", "탐지지표", "apt", "ddos",
            "ids", "ips", "bcp", "drp", "isms-p"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question_lower)
        
        # 문장 길이
        length = len(question)
        
        # 난이도 계산
        if term_count >= 3 or length > 300:
            return "고급"
        elif term_count >= 1 or length > 150:
            return "중급"
        else:
            return "초급"
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화 (강화)"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 숫자만 추출하고 범위 검증
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            # 유효한 답변이 없으면 빈 문자열 반환
            return ""
        
        else:
            # 주관식 답변 한국어 정리 (더 강화)
            answer = self.clean_korean_text(answer)
            
            # 의미 없는 짧은 문장 제거
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 길이 제한
            if len(answer) > self.korean_requirements["max_length"]:
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:3])
                if len(answer) > self.korean_requirements["max_length"]:
                    answer = answer[:self.korean_requirements["max_length"]]
            
            # 마침표 확인
            if answer and not answer.endswith(('.', '다', '요', '함')):
                answer += "."
            
            return answer
    
    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화 (한국어 전용)"""
        return self.normalize_korean_answer(answer, question_type, max_choice)
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환 (강화)"""
        total = max(self.processing_stats["total_processed"], 1)
        intent_total = max(self.processing_stats["intent_analysis_accuracy"]["total"], 1)
        intent_match_total = max(self.processing_stats["intent_match_accuracy"]["total"], 1)
        
        return {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "choice_count_errors": self.processing_stats["choice_count_errors"],
            "intent_analysis_accuracy_rate": (self.processing_stats["intent_analysis_accuracy"]["correct"] / intent_total) * 100,
            "intent_match_accuracy_rate": (self.processing_stats["intent_match_accuracy"]["correct"] / intent_match_total) * 100,  # 신규
            "pattern_matching_avg": self.processing_stats["pattern_matching_score"] / max(intent_total, 1),
            "semantic_analysis_avg": self.processing_stats["semantic_analysis_score"] / max(intent_total, 1),
            "context_understanding_avg": self.processing_stats["context_understanding_score"] / max(intent_total, 1),
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats["question_type_accuracy"],
            # 정확한 의도 일치 통계 제공
            "intent_match_accuracy": {
                "correct": self.processing_stats["intent_match_accuracy"]["correct"],
                "total": self.processing_stats["intent_match_accuracy"]["total"]
            }
        }
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()