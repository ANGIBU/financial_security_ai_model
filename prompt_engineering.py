# prompt_engineering.py

import re
import hashlib
from typing import Dict, List, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_enhanced_korean_templates()
        self.examples = self._build_comprehensive_korean_examples()
        
        self.prompt_cache = {}
        self.template_cache = {}
        self.max_cache_size = 500
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {},
            "success_rate_by_template": {},
            "failure_analysis": {}
        }
        
        self.dynamic_templates = self._build_dynamic_templates()
        self.failure_recovery_templates = self._build_failure_recovery_templates()
        
    def _build_enhanced_korean_templates(self) -> Dict[str, str]:
        templates = {}
        
        templates["mc_direct"] = """### 문제 분석
{question}

위 문제를 신중히 분석하고 정답 번호만 출력하세요.
정답:"""

        templates["mc_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 문제
{question}

### 답변 규칙
1. 반드시 1, 2, 3, 4, 5 중 하나만 선택
2. 정답 번호만 출력
3. 설명 없이 숫자만

정답:"""

        templates["mc_enhanced_negative"] = """당신은 한국의 금융보안 전문가입니다.

### 부정형 문제 분석
{question}

### 중요 분석 포인트
이 문제는 '{keyword}' 문제입니다.
- 틀린 것, 해당하지 않는 것, 적절하지 않은 것을 찾으세요
- 부정형 표현에 주의하여 신중히 판단하세요

### 답변 규칙
1. 반드시 1, 2, 3, 4, 5 중 하나만 선택
2. 부정형 문제임을 명심하고 분석
3. 정답 번호만 출력

정답:"""

        templates["mc_financial_specialized"] = """### 금융투자업 분류 문제
{question}

### 금융업 분류 핵심 지식
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 금융투자업 아님: 소비자금융업, 보험중개업, 은행업
- 자본시장법에 따른 엄격한 구분 적용

### 분석 포인트
- 소비자금융업은 개인 대상 소액 신용대출 업무
- 보험중개업은 보험 계약 중개 업무
- 둘 다 금융투자업에 해당하지 않음

정답:"""

        templates["mc_risk_management_specialized"] = """### 위험관리 계획 수립 문제
{question}

### 위험관리 핵심 지식
- 위험관리 계획 수립 시 고려요소: 위험 대상, 관리 기간, 수행 인력
- 위험 대응 전략: 위험 회피, 위험 완화, 위험 전가, 위험 수용
- 위험수용은 대응전략의 하나이지 별도 고려요소가 아님

### 분석 포인트
- 위험수용은 위험 대응 방법 중 하나
- 계획 수립 시 고려요소와 대응전략을 구분해야 함

정답:"""

        templates["mc_management_system_specialized"] = """### 관리체계 구축 문제
{question}

### 관리체계 수립 핵심 지식
- 정책수립 단계에서 가장 중요: 경영진의 참여와 지원
- 그 다음 단계: 최고책임자 지정, 조직 구성, 자원 할당
- 경영진 참여 없이는 실효성 있는 정책 수립 불가능

### 분석 포인트
- 정보보호 및 개인정보보호 정책의 제정과 개정
- 최고경영진의 의지와 지원이 성공의 핵심
- 정책 수립이 모든 관리체계의 기반

정답:"""

        templates["mc_disaster_recovery_specialized"] = """### 재해복구 계획 수립 문제
{question}

### 재해복구 계획 핵심 지식
- 재해복구 계획 포함 사항: 복구 절차, 비상연락체계, 복구목표시간
- 업무연속성 관리: BCP, 핵심 업무 식별, 대체 방안
- 개인정보 파기 절차는 재해복구와 직접적 관련 없음

### 분석 포인트
- 재해복구는 재해 발생 시 신속한 업무 복구가 목적
- 개인정보 파기는 일상적 개인정보 관리 업무
- 재해복구 계획과는 별개의 영역

정답:"""

        templates["mc_cyber_security_specialized"] = """### 사이버보안 문제
{question}

### 사이버보안 핵심 지식
- 트로이 목마: 정상 프로그램으로 위장한 악성코드
- RAT (원격 접근 트로이): 시스템을 원격으로 제어
- 주요 탐지 지표: 비정상적 네트워크 연결, 시스템 리소스 증가
- 시스템 감염 증상: 알 수 없는 프로세스, 방화벽 규칙 변경

### 분석 포인트
- 원격 제어 기능이 핵심 특징
- 시스템 자원 사용량 급증
- 네트워크 트래픽 이상 패턴

정답:"""

        templates["mc_encryption_specialized"] = """### 암호화 기술 문제
{question}

### 암호화 핵심 지식
- 대칭키 암호화: 빠른 처리, 같은 키로 암호화/복호화
- 공개키 암호화: 안전한 키 교환, 디지털 서명 가능
- 해시 함수: 무결성 검증, 단방향 암호화
- 키 관리: 생성, 배포, 저장, 폐기의 전 과정 관리

### 분석 포인트
- 용도에 따른 암호화 방식 선택
- 키 관리의 중요성
- 성능과 보안의 균형

정답:"""

        templates["mc_access_control_specialized"] = """### 접근제어 문제
{question}

### 접근제어 핵심 지식
- 접근매체: 전자금융거래에서 이용자 인증 수단
- 금융회사 의무: 안전하고 신뢰할 수 있는 접근매체 선정
- 다중인증: 2개 이상 인증요소 조합 (지식, 소유, 특성)
- 생체인증: 지문, 홍채, 얼굴 등 생체 정보 활용

### 분석 포인트
- 접근매체의 안전성과 신뢰성
- 인증 강도와 편의성의 균형
- 법적 요구사항 준수

정답:"""

        templates["subj_enhanced"] = """당신은 한국의 금융보안 전문가입니다.

### 중요 규칙
1. 반드시 순수 한국어로만 답변
2. 한자, 영어 등 외국어 절대 금지
3. 100-400자 내외로 답변
4. 전문적이고 명확한 한국어 사용

### 질문
{question}

### 답변 형식
관련 법령과 규정에 따라 구체적이고 체계적으로 설명하되,
순수 한국어만 사용하여 전문적으로 답변하세요.

답변:"""

        templates["subj_trojan_specialized"] = """당신은 한국의 사이버보안 전문가입니다.

### 중요 규칙
1. 반드시 순수 한국어로만 답변
2. 한자, 영어 등 외국어 절대 금지
3. 150-350자 내외로 답변

### 질문
{question}

### 트로이 목마 전문 지식
- 트로이 목마: 정상 프로그램으로 위장한 악성코드
- 원격 접근 트로이 목마(RAT): 원격 시스템 제어 기능
- 주요 탐지 지표: 비정상 네트워크 연결, 리소스 사용 증가
- 감염 증상: 알 수 없는 프로세스, 시스템 설정 변경

### 답변 구조
1. 트로이 목마의 정의와 특징
2. 원격 접근 트로이 목마의 기능
3. 주요 탐지 지표와 대응 방안

답변:"""

        templates["subj_personal_info_specialized"] = """당신은 한국의 개인정보보호 전문가입니다.

### 질문
{question}

### 개인정보보호 전문 지식
- 개인정보보호법: 개인정보 처리의 기본 원칙과 절차
- 정보주체 권리: 열람, 정정, 삭제, 처리정지 요구권
- 안전성 확보조치: 기술적, 관리적, 물리적 보호대책
- 개인정보 유출 시: 지체 없는 통지와 신고 의무

### 답변 지침
개인정보보호법에 따른 구체적인 조치사항을 순수 한국어로 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""

        templates["subj_electronic_specialized"] = """당신은 한국의 전자금융 전문가입니다.

### 질문
{question}

### 전자금융 전문 지식
- 전자금융거래법: 전자적 장치를 통한 금융거래 규율
- 접근매체: 이용자 및 거래 진실성 확보 수단
- 안전성 확보: 암호화, 접근통제, 거래 모니터링
- 이용자 보호: 손해배상, 오류정정, 거래내역 통지

### 답변 지침
전자금융거래법에 따른 안전성 확보 방안을 순수 한국어로 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""

        templates["subj_risk_management_specialized"] = """당신은 한국의 위험관리 전문가입니다.

### 질문
{question}

### 위험관리 전문 지식
- 위험관리 체계: 식별, 분석, 평가, 대응, 모니터링
- 위험 대응 전략: 회피, 완화, 전가, 수용
- 위험 평가: 발생 가능성과 영향도 분석
- 모니터링: 지속적 위험 상황 감시

### 답변 지침
위험관리 체계와 대응전략을 순수 한국어로 설명하세요.
위험 식별, 평가, 대응, 모니터링 과정을 포함하여 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""

        templates["subj_management_system_specialized"] = """당신은 한국의 관리체계 전문가입니다.

### 질문
{question}

### 관리체계 전문 지식
- 관리체계 구성요소: 정책, 조직, 절차, 기술
- 정책 수립: 경영진 참여, 목표 설정, 책임 할당
- 운영 관리: 지속적 모니터링, 정기적 점검
- 개선 활동: 내부 감사, 경영 검토, 시정 조치

### 답변 지침
관리체계 수립과 운영 방안을 순수 한국어로 설명하세요.
정책 수립, 조직 구성, 역할 분담, 지속적 개선을 포함하여 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""

        templates["subj_incident_response_specialized"] = """당신은 한국의 사고대응 전문가입니다.

### 질문
{question}

### 사고대응 전문 지식
- 사고대응 단계: 준비, 탐지, 분석, 억제, 제거, 복구
- 대응팀 구성: 사고대응팀, 기술팀, 의사소통팀
- 증거 수집: 로그 분석, 포렌식 조사
- 복구 계획: 시스템 복구, 업무 연속성 확보

### 답변 지침
침해사고 대응 절차와 복구 방안을 순수 한국어로 설명하세요.
사고 탐지, 분석, 대응, 복구, 사후관리 단계를 포함하여 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""

        templates["subj_crypto_specialized"] = """당신은 한국의 암호화 전문가입니다.

### 질문
{question}

### 암호화 전문 지식
- 암호화 기술: 대칭키, 공개키, 해시 함수
- 키 관리: 생성, 배포, 저장, 갱신, 폐기
- 암호화 적용: 전송 구간, 저장 데이터
- 성능 고려: 암호화 부하, 처리 속도

### 답변 지침
암호화 기술과 키 관리 방안을 순수 한국어로 설명하세요.
대칭키, 공개키 암호화와 해시 함수 활용을 포함하여 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""

        templates["subj_law_compliance_specialized"] = """당신은 한국의 금융법규 준수 전문가입니다.

### 질문
{question}

### 법규 준수 전문 지식
- 관련 법령: 개인정보보호법, 전자금융거래법, 정보통신망법
- 준수 의무: 법적 요구사항, 규제 기관 가이드라인
- 점검 체계: 정기 점검, 내부 감사, 외부 평가
- 위반 시 제재: 과태료, 과징금, 업무정지

### 답변 지침
관련 법령의 준수 사항과 의무 사항을 순수 한국어로 설명하세요.
법적 근거와 구체적 조치 방안을 포함하여 설명하세요.
100-350자 내외, 외국어 사용 금지

답변:"""
        
        return templates
    
    def _build_dynamic_templates(self) -> Dict[str, str]:
        return {
            "adaptive_mc": """### 문제 유형: {question_type_analysis}
{question}

### 적응형 분석 접근법
{adaptive_strategy}

### 핵심 판단 포인트
{key_decision_points}

정답:""",
            
            "context_aware_subj": """### 맥락 인식 답변
{question}

### 문맥 분석 결과
{context_analysis}

### 전문가 답변 전략
{expert_strategy}

### 순수 한국어 답변 요구사항
- 한자, 영어 절대 금지
- 전문적이고 정확한 표현
- {target_length}자 내외

답변:""",
            
            "failure_adaptive": """### 재시도 최적화 답변
{question}

### 이전 실패 분석
{failure_analysis}

### 개선된 접근 방법
{improved_approach}

### 강화된 답변 전략
{enhanced_strategy}

정답:"""
        }
    
    def _build_failure_recovery_templates(self) -> Dict[str, str]:
        return {
            "mc_extraction_failure": """### 명확한 객관식 답변 요구
{question}

### 절대 준수 사항
1. 반드시 1, 2, 3, 4, 5 중 정확히 하나만 선택
2. 다른 텍스트 없이 숫자만 출력
3. 추가 설명이나 분석 금지

예시: 3

지금 정답 번호만 출력하세요:""",
            
            "korean_quality_failure": """### 순수 한국어 답변 재요청
{question}

### 엄격한 언어 요구사항
- 한국어만 사용 (한자, 영어, 기타 외국어 절대 금지)
- 전문 용어는 한국어로 표현
- 자연스럽고 정확한 한국어

### 개선된 답변 요청
위 질문에 대해 순수 한국어로만 전문적으로 답변하세요.

답변:""",
            
            "length_adjustment": """### 답변 길이 조정 요청
{question}

### 길이 요구사항
- 목표 길이: {target_length}자
- 현재 문제: {length_issue}
- 핵심 내용 중심으로 {adjustment_direction}

### 조정된 답변 요청
위 길이 요구사항에 맞춰 다시 답변하세요.

답변:"""
        }
    
    def _build_comprehensive_korean_examples(self) -> Dict[str, Dict]:
        examples = {
            "mc_financial": {
                "question": "다음 중 금융투자업의 구분에 해당하지 않는 것은?",
                "answer": "1",
                "reasoning": "소비자금융업은 금융투자업이 아님",
                "domain": "금융법규"
            },
            "mc_risk": {
                "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?",
                "answer": "2",
                "reasoning": "위험 수용은 대응 전략의 하나",
                "domain": "위험관리"
            },
            "mc_management": {
                "question": "관리체계 수립 시 정책수립 단계에서 가장 중요한 것은?",
                "answer": "2",
                "reasoning": "경영진의 참여와 지원이 가장 중요",
                "domain": "관리체계"
            },
            "mc_recovery": {
                "question": "재해복구 계획 수립 시 고려사항으로 옳지 않은 것은?",
                "answer": "3",
                "reasoning": "개인정보 파기 절차는 재해복구와 직접 관련 없음",
                "domain": "재해복구"
            },
            "mc_personal": {
                "question": "개인정보의 정의로 가장 적절한 것은?",
                "answer": "2",
                "reasoning": "살아있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보",
                "domain": "개인정보보호"
            },
            "mc_electronic": {
                "question": "전자금융거래의 정의로 가장 적절한 것은?",
                "answer": "2",
                "reasoning": "전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래",
                "domain": "전자금융"
            },
            "subj_trojan": {
                "question": "트로이 목마 기반 원격제어 악성코드의 특징과 탐지 지표를 설명하세요.",
                "answer": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행, 방화벽 규칙 변경, 레지스트리 변경 등이 있습니다.",
                "domain": "사이버보안"
            },
            "subj_privacy": {
                "question": "개인정보 유출 시 조치사항을 설명하세요.",
                "answer": "개인정보 유출 시 개인정보보호법에 따라 지체 없이 정보주체에게 통지하고, 일정 규모 이상의 유출 시 개인정보보호위원회에 신고해야 합니다. 유출 통지 내용에는 유출 항목, 시점, 경위, 피해 최소화 방법, 담당부서 연락처 등이 포함되어야 합니다.",
                "domain": "개인정보보호"
            }
        }
        return examples
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Dict, structure: Dict) -> str:
        
        cache_key = hash(f"{question[:100]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        if question_type == "multiple_choice":
            prompt = self._create_adaptive_mc_prompt(question, analysis, structure)
        else:
            prompt = self._create_adaptive_subj_prompt(question, analysis, structure)
        
        if len(self.prompt_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]
        self.prompt_cache[cache_key] = prompt
        
        self._update_stats(analysis)
        
        return prompt
    
    def _create_adaptive_mc_prompt(self, question: str, analysis: Dict, structure: Dict) -> str:
        question_lower = question.lower()
        domains = analysis.get("domain", [])
        
        if "금융투자업" in question_lower:
            if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                prompt = self.templates["mc_financial_specialized"].format(question=question)
                self.stats["template_usage"]["mc_financial_specialized"] = self.stats["template_usage"].get("mc_financial_specialized", 0) + 1
                return prompt
        
        if "위험" in question_lower and "관리" in question_lower and "계획" in question_lower:
            if "위험수용" in question_lower or "위험 수용" in question_lower:
                prompt = self.templates["mc_risk_management_specialized"].format(question=question)
                self.stats["template_usage"]["mc_risk_management_specialized"] = self.stats["template_usage"].get("mc_risk_management_specialized", 0) + 1
                return prompt
        
        if "관리체계" in question_lower and "정책" in question_lower:
            if "경영진" in question_lower or "가장중요" in question_lower:
                prompt = self.templates["mc_management_system_specialized"].format(question=question)
                self.stats["template_usage"]["mc_management_system_specialized"] = self.stats["template_usage"].get("mc_management_system_specialized", 0) + 1
                return prompt
        
        if "재해복구" in question_lower or "재해 복구" in question_lower:
            if "개인정보파기" in question_lower or "개인정보 파기" in question_lower:
                prompt = self.templates["mc_disaster_recovery_specialized"].format(question=question)
                self.stats["template_usage"]["mc_disaster_recovery_specialized"] = self.stats["template_usage"].get("mc_disaster_recovery_specialized", 0) + 1
                return prompt
        
        if "트로이" in question_lower or "악성코드" in question_lower or "원격" in question_lower:
            prompt = self.templates["mc_cyber_security_specialized"].format(question=question)
            self.stats["template_usage"]["mc_cyber_security_specialized"] = self.stats["template_usage"].get("mc_cyber_security_specialized", 0) + 1
            return prompt
        
        if "암호화" in question_lower or "암호" in question_lower or "키관리" in question_lower:
            prompt = self.templates["mc_encryption_specialized"].format(question=question)
            self.stats["template_usage"]["mc_encryption_specialized"] = self.stats["template_usage"].get("mc_encryption_specialized", 0) + 1
            return prompt
        
        if "접근매체" in question_lower or "접근제어" in question_lower or "다중인증" in question_lower:
            prompt = self.templates["mc_access_control_specialized"].format(question=question)
            self.stats["template_usage"]["mc_access_control_specialized"] = self.stats["template_usage"].get("mc_access_control_specialized", 0) + 1
            return prompt
        
        if structure.get("has_all_option", False):
            for choice_line in question.split('\n'):
                if re.match(r'^\s*[5]', choice_line):
                    if "모두" in choice_line or "전부" in choice_line:
                        prompt = self._create_all_option_prompt(question)
                        self.stats["template_usage"]["mc_all_option"] = self.stats["template_usage"].get("mc_all_option", 0) + 1
                        return prompt
        
        if structure.get("has_negative", False):
            negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
            keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
            
            prompt = self.templates["mc_enhanced_negative"].format(
                question=question,
                keyword=keyword
            )
            self.stats["template_usage"]["mc_enhanced_negative"] = self.stats["template_usage"].get("mc_enhanced_negative", 0) + 1
        else:
            prompt = self.templates["mc_direct"].format(question=question)
            self.stats["template_usage"]["mc_direct"] = self.stats["template_usage"].get("mc_direct", 0) + 1
        
        return prompt
    
    def _create_adaptive_subj_prompt(self, question: str, analysis: Dict, structure: Dict) -> str:
        question_lower = question.lower()
        domains = analysis.get("domain", [])
        
        if "트로이" in question_lower and ("악성코드" in question_lower or "원격" in question_lower or "탐지" in question_lower):
            prompt = self.templates["subj_trojan_specialized"].format(question=question)
            self.stats["template_usage"]["subj_trojan_specialized"] = self.stats["template_usage"].get("subj_trojan_specialized", 0) + 1
        elif "개인정보보호" in domains or "개인정보" in question_lower:
            prompt = self.templates["subj_personal_info_specialized"].format(question=question)
            self.stats["template_usage"]["subj_personal_info_specialized"] = self.stats["template_usage"].get("subj_personal_info_specialized", 0) + 1
        elif "전자금융" in domains or "전자금융" in question_lower:
            prompt = self.templates["subj_electronic_specialized"].format(question=question)
            self.stats["template_usage"]["subj_electronic_specialized"] = self.stats["template_usage"].get("subj_electronic_specialized", 0) + 1
        elif "위험관리" in domains or ("위험" in question_lower and "관리" in question_lower):
            prompt = self.templates["subj_risk_management_specialized"].format(question=question)
            self.stats["template_usage"]["subj_risk_management_specialized"] = self.stats["template_usage"].get("subj_risk_management_specialized", 0) + 1
        elif "관리체계" in domains or ("관리체계" in question_lower and "정책" in question_lower):
            prompt = self.templates["subj_management_system_specialized"].format(question=question)
            self.stats["template_usage"]["subj_management_system_specialized"] = self.stats["template_usage"].get("subj_management_system_specialized", 0) + 1
        elif "사고대응" in domains or ("사고" in question_lower and ("대응" in question_lower or "복구" in question_lower)):
            prompt = self.templates["subj_incident_response_specialized"].format(question=question)
            self.stats["template_usage"]["subj_incident_response_specialized"] = self.stats["template_usage"].get("subj_incident_response_specialized", 0) + 1
        elif "암호화" in domains or ("암호" in question_lower and "키" in question_lower):
            prompt = self.templates["subj_crypto_specialized"].format(question=question)
            self.stats["template_usage"]["subj_crypto_specialized"] = self.stats["template_usage"].get("subj_crypto_specialized", 0) + 1
        elif "법령" in question_lower or "규정" in question_lower or "의무" in question_lower:
            prompt = self.templates["subj_law_compliance_specialized"].format(question=question)
            self.stats["template_usage"]["subj_law_compliance_specialized"] = self.stats["template_usage"].get("subj_law_compliance_specialized", 0) + 1
        else:
            prompt = self.templates["subj_enhanced"].format(question=question)
            self.stats["template_usage"]["subj_enhanced"] = self.stats["template_usage"].get("subj_enhanced", 0) + 1
        
        return prompt
    
    def _create_all_option_prompt(self, question: str) -> str:
        return f"""### 모든 선택지 포함 문제
{question}

### 분석 포인트
- 마지막 선택지에 '모두' 또는 '전부'가 포함된 문제
- 모든 조건을 만족하는지 신중히 검토
- 부분적 만족이 아닌 완전한 만족 여부 판단

### 판단 기준
1. 앞의 모든 선택지가 조건에 해당하는가?
2. 예외나 제외 사항은 없는가?
3. 문제에서 요구하는 조건을 모두 충족하는가?

정답:"""
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        question_lower = question.lower()
        
        if question_type == "multiple_choice":
            return self._create_reinforced_mc_prompt(question, question_lower)
        else:
            return self._create_reinforced_subj_prompt(question, question_lower)
    
    def _create_reinforced_mc_prompt(self, question: str, question_lower: str) -> str:
        if "금융투자업" in question_lower:
            if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                return self.templates["mc_financial_specialized"].format(question=question)
        
        if "위험" in question_lower and "관리" in question_lower:
            if "위험수용" in question_lower or "위험 수용" in question_lower:
                return self.templates["mc_risk_management_specialized"].format(question=question)
        
        if "관리체계" in question_lower and "정책" in question_lower:
            if "경영진" in question_lower or "참여" in question_lower:
                return self.templates["mc_management_system_specialized"].format(question=question)
        
        if "재해" in question_lower and "복구" in question_lower:
            if "개인정보" in question_lower and "파기" in question_lower:
                return self.templates["mc_disaster_recovery_specialized"].format(question=question)
        
        if "트로이" in question_lower or "악성코드" in question_lower:
            return self.templates["mc_cyber_security_specialized"].format(question=question)
        
        if "암호화" in question_lower or "암호" in question_lower:
            return self.templates["mc_encryption_specialized"].format(question=question)
        
        if "접근매체" in question_lower or "접근제어" in question_lower:
            return self.templates["mc_access_control_specialized"].format(question=question)
        
        for choice_line in question.split('\n'):
            if re.match(r'^\s*[5]', choice_line):
                if "모두" in choice_line or "전부" in choice_line:
                    return self._create_all_option_prompt(question)
        
        if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
            negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
            keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
            return self.templates["mc_enhanced_negative"].format(question=question, keyword=keyword)
        
        return self.templates["mc_direct"].format(question=question)
    
    def _create_reinforced_subj_prompt(self, question: str, question_lower: str) -> str:
        if "트로이" in question_lower and any(word in question_lower for word in ["악성코드", "원격", "rat", "탐지"]):
            return self.templates["subj_trojan_specialized"].format(question=question)
        elif "개인정보" in question_lower:
            return self.templates["subj_personal_info_specialized"].format(question=question)
        elif "전자금융" in question_lower:
            return self.templates["subj_electronic_specialized"].format(question=question)
        elif "위험" in question_lower and "관리" in question_lower:
            return self.templates["subj_risk_management_specialized"].format(question=question)
        elif "관리체계" in question_lower:
            return self.templates["subj_management_system_specialized"].format(question=question)
        elif "사고" in question_lower and ("대응" in question_lower or "복구" in question_lower):
            return self.templates["subj_incident_response_specialized"].format(question=question)
        elif "암호" in question_lower:
            return self.templates["subj_crypto_specialized"].format(question=question)
        elif "법령" in question_lower or "규정" in question_lower:
            return self.templates["subj_law_compliance_specialized"].format(question=question)
        else:
            return self.templates["subj_enhanced"].format(question=question)
    
    def create_failure_recovery_prompt(self, question: str, question_type: str, 
                                     failure_type: str, previous_response: str = "") -> str:
        
        if failure_type == "extraction_failure" and question_type == "multiple_choice":
            return self.failure_recovery_templates["mc_extraction_failure"].format(question=question)
        
        elif failure_type == "korean_quality_failure":
            return self.failure_recovery_templates["korean_quality_failure"].format(question=question)
        
        elif failure_type == "length_issue":
            if len(previous_response) < 50:
                target_length = "100-200"
                length_issue = "답변이 너무 짧음"
                adjustment_direction = "내용을 확장하여"
            else:
                target_length = "150-300"
                length_issue = "답변이 너무 김"
                adjustment_direction = "핵심 내용만으로 간소화하여"
            
            return self.failure_recovery_templates["length_adjustment"].format(
                question=question,
                target_length=target_length,
                length_issue=length_issue,
                adjustment_direction=adjustment_direction
            )
        
        else:
            return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_few_shot_prompt(self, question: str, question_type: str,
                             analysis: Dict, num_examples: int = 1) -> str:
        
        prompt_parts = ["다음은 한국어 금융보안 문제 예시입니다.\n"]
        
        if question_type == "multiple_choice":
            question_lower = question.lower()
            
            if "금융투자업" in question_lower:
                example = self.examples["mc_financial"]
            elif "위험" in question_lower and "관리" in question_lower:
                example = self.examples["mc_risk"]
            elif "관리체계" in question_lower:
                example = self.examples["mc_management"]
            elif "재해복구" in question_lower:
                example = self.examples["mc_recovery"]
            elif "개인정보" in question_lower and "정의" in question_lower:
                example = self.examples["mc_personal"]
            elif "전자금융" in question_lower and "정의" in question_lower:
                example = self.examples["mc_electronic"]
            else:
                example = self.examples["mc_financial"]
            
            prompt_parts.append(f"예시 문제: {example['question']}")
            prompt_parts.append(f"정답: {example['answer']}")
            prompt_parts.append(f"논리: {example['reasoning']}\n")
        else:
            if "트로이" in question:
                example = self.examples["subj_trojan"]
                prompt_parts.append(f"예시 질문: {example['question']}")
                prompt_parts.append(f"답변: {example['answer']}\n")
            elif "개인정보" in question and "유출" in question:
                example = self.examples["subj_privacy"]
                prompt_parts.append(f"예시 질문: {example['question']}")
                prompt_parts.append(f"답변: {example['answer']}\n")
        
        prompt_parts.append(f"현재 문제:\n{question}")
        prompt_parts.append("위 예시처럼 답하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def create_dynamic_prompt(self, question: str, question_type: str,
                            analysis: Dict, structure: Dict, context: Dict) -> str:
        
        if question_type == "multiple_choice":
            question_type_analysis = self._analyze_mc_question_type(question, structure)
            adaptive_strategy = self._get_adaptive_mc_strategy(question, analysis)
            key_decision_points = self._extract_mc_decision_points(question, structure)
            
            return self.dynamic_templates["adaptive_mc"].format(
                question_type_analysis=question_type_analysis,
                question=question,
                adaptive_strategy=adaptive_strategy,
                key_decision_points=key_decision_points
            )
        else:
            context_analysis = self._analyze_subjective_context(question, analysis)
            expert_strategy = self._get_expert_strategy(question, analysis)
            target_length = self._determine_target_length(question, structure)
            
            return self.dynamic_templates["context_aware_subj"].format(
                question=question,
                context_analysis=context_analysis,
                expert_strategy=expert_strategy,
                target_length=target_length
            )
    
    def _analyze_mc_question_type(self, question: str, structure: Dict) -> str:
        analysis_parts = []
        
        if structure.get("has_negative", False):
            analysis_parts.append("부정형 문제 (틀린 것, 해당하지 않는 것 찾기)")
        
        if structure.get("has_all_option", False):
            analysis_parts.append("전체 포함 문제 (모두 해당하는지 판단)")
        
        if "정의" in question:
            analysis_parts.append("정의 문제 (개념과 의미 이해)")
        
        if "가장" in question and "중요" in question:
            analysis_parts.append("우선순위 문제 (중요도 판단)")
        
        return " / ".join(analysis_parts) if analysis_parts else "일반 객관식 문제"
    
    def _get_adaptive_mc_strategy(self, question: str, analysis: Dict) -> str:
        domains = analysis.get("domain", [])
        strategies = []
        
        if "개인정보보호" in domains:
            strategies.append("개인정보보호법 조항과 원칙 적용")
        
        if "전자금융" in domains:
            strategies.append("전자금융거래법 요구사항 검토")
        
        if "정보보안" in domains:
            strategies.append("정보보호관리체계 기준 적용")
        
        if not strategies:
            strategies.append("금융보안 일반 원칙 적용")
        
        return " + ".join(strategies)
    
    def _extract_mc_decision_points(self, question: str, structure: Dict) -> str:
        points = []
        
        if "금융투자업" in question:
            points.append("• 소비자금융업과 보험중개업은 금융투자업이 아님")
        
        if "위험관리" in question:
            points.append("• 위험수용은 대응전략이지 별도 고려요소가 아님")
        
        if "관리체계" in question:
            points.append("• 정책수립에서 경영진 참여가 가장 중요")
        
        if "재해복구" in question:
            points.append("• 개인정보 파기는 재해복구와 무관")
        
        if not points:
            points.append("• 문제의 핵심 키워드와 조건 정확히 파악")
            points.append("• 선택지별 적합성 신중히 검토")
        
        return "\n".join(points)
    
    def _analyze_subjective_context(self, question: str, analysis: Dict) -> str:
        domains = analysis.get("domain", [])
        context_parts = []
        
        if "개인정보보호" in domains:
            context_parts.append("개인정보보호 법령 준수 관점")
        
        if "전자금융" in domains:
            context_parts.append("전자금융 안전성 확보 관점")
        
        if "정보보안" in domains:
            context_parts.append("정보보안 관리체계 관점")
        
        if "사이버보안" in domains:
            context_parts.append("사이버 위협 대응 관점")
        
        return " + ".join(context_parts) if context_parts else "금융보안 종합 관점"
    
    def _get_expert_strategy(self, question: str, analysis: Dict) -> str:
        if "설명하세요" in question:
            return "체계적 설명: 정의 → 특징 → 방법 → 예시"
        elif "방안" in question or "대책" in question:
            return "해결방안 제시: 현황 → 문제점 → 구체적 방안 → 기대효과"
        elif "절차" in question or "과정" in question:
            return "단계별 절차: 준비 → 실행 → 점검 → 개선"
        else:
            return "전문가 답변: 법적근거 → 핵심내용 → 실무적용"
    
    def _determine_target_length(self, question: str, structure: Dict) -> str:
        if "간단히" in question or "요약" in question:
            return "100-200"
        elif "상세히" in question or "구체적으로" in question:
            return "250-400"
        else:
            return "150-300"
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        korean_prefix = "### 중요: 반드시 한국어로만 답변하세요 ###\n\n"
        
        if "solar" in model_name.lower():
            optimized = f"{korean_prefix}### User:\n{prompt}\n\n### Assistant:\n"
        elif "llama" in model_name.lower():
            optimized = f"{korean_prefix}<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            optimized = korean_prefix + prompt
        
        return optimized
    
    def track_template_performance(self, template_name: str, success: bool, confidence: float):
        if template_name not in self.stats["success_rate_by_template"]:
            self.stats["success_rate_by_template"][template_name] = []
        
        self.stats["success_rate_by_template"][template_name].append({
            "success": success,
            "confidence": confidence
        })
        
        if len(self.stats["success_rate_by_template"][template_name]) > 50:
            self.stats["success_rate_by_template"][template_name] = \
                self.stats["success_rate_by_template"][template_name][-50:]
    
    def analyze_failure_patterns(self, question: str, question_type: str, 
                                failed_response: str, failure_type: str):
        failure_key = f"{question_type}_{failure_type}"
        
        if failure_key not in self.stats["failure_analysis"]:
            self.stats["failure_analysis"][failure_key] = []
        
        self.stats["failure_analysis"][failure_key].append({
            "question_sample": question[:100],
            "response_sample": failed_response[:100],
            "failure_type": failure_type
        })
        
        if len(self.stats["failure_analysis"][failure_key]) > 20:
            self.stats["failure_analysis"][failure_key] = \
                self.stats["failure_analysis"][failure_key][-20:]
    
    def _update_stats(self, analysis: Dict):
        domains = analysis.get("domain", ["일반"])
        for domain in domains:
            self.stats["domain_distribution"][domain] = self.stats["domain_distribution"].get(domain, 0) + 1
    
    def get_performance_report(self) -> Dict:
        total_prompts = sum(self.stats["template_usage"].values())
        
        template_performance = {}
        for template_name, performance_data in self.stats["success_rate_by_template"].items():
            if len(performance_data) >= 3:
                success_rate = sum(1 for p in performance_data if p["success"]) / len(performance_data)
                avg_confidence = sum(p["confidence"] for p in performance_data) / len(performance_data)
                template_performance[template_name] = {
                    "success_rate": success_rate,
                    "avg_confidence": avg_confidence,
                    "sample_size": len(performance_data)
                }
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "template_usage": self.stats["template_usage"],
            "domain_distribution": self.stats["domain_distribution"],
            "template_performance": template_performance,
            "failure_patterns": {k: len(v) for k, v in self.stats["failure_analysis"].items()}
        }
    
    def cleanup(self):
        performance_report = self.get_performance_report()
        if performance_report["total_prompts"] > 0:
            best_template = max(self.stats["template_usage"].items(), key=lambda x: x[1])
            if best_template[1] > 0:
                pass
        
        self.prompt_cache.clear()
        self.template_cache.clear()