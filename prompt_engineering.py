# prompt_engineering.py
"""
프롬프트 엔지니어링 시스템
"""

import re
import hashlib
from typing import Dict, List, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    """프롬프트 엔지니어링 클래스 - 한국어 특화"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_korean_templates()
        self.examples = self._build_korean_examples()
        
        # 캐시
        self.prompt_cache = {}
        self.template_cache = {}
        
        # 통계
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {}
        }
    
    def _build_korean_templates(self) -> Dict[str, str]:
        """한국어 전용 템플릿"""
        templates = {}
        
        # 객관식 기본 템플릿
        templates["mc_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 오직 한국어로만 답변하세요
2. 한자(漢字), 영어(English), 일본어, 중국어 등 외국어 사용 금지
3. 답변은 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 선택

### 문제
{question}

### 지시사항
위 문제를 읽고 정답 번호만 출력하세요.
추가 설명이나 분석은 필요 없습니다.

정답:"""

        # 객관식 부정형 템플릿  
        templates["mc_negative"] = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 오직 한국어로만 답변하세요
2. 외국어 사용 절대 금지
3. 숫자 1~5 중 하나만 선택

### 문제
{question}

### 주의사항
이 문제는 '{keyword}'를 찾는 문제입니다.
틀리거나 해당하지 않는 것을 선택하세요.

정답:"""

        # 주관식 기본 템플릿
        templates["subj_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 반드시 한국어로만 답변하세요
2. 한자(中文), 영어(English) 등 외국어 절대 사용 금지
3. 전문적이고 명확한 한국어 사용
4. 80자 이상 600자 이내로 답변

### 질문
{question}

### 답변 형식
관련 법령과 규정에 따라 구체적으로 설명하되,
순수 한국어만 사용하여 답변하세요.

답변:"""

        # 도메인별 주관식 템플릿
        templates["subj_personal_info"] = """당신은 한국의 개인정보보호 전문가입니다.

### 절대 규칙
1. 순수 한국어만 사용
2. 외국어 절대 금지
3. 개인정보보호법 중심 답변

### 질문
{question}

### 답변 지침
개인정보보호법에 따른 구체적 방안을 한국어로만 설명하세요.

답변:"""

        templates["subj_electronic"] = """당신은 한국의 전자금융 전문가입니다.

### 절대 규칙
1. 순수 한국어만 사용
2. 외국어 절대 금지
3. 전자금융거래법 중심 답변

### 질문
{question}

### 답변 지침
전자금융거래법에 따른 안전한 거래 방안을 한국어로만 설명하세요.

답변:"""

        templates["subj_security"] = """당신은 한국의 정보보안 전문가입니다.

### 절대 규칙
1. 순수 한국어만 사용
2. 외국어 절대 금지
3. 정보보안 관리체계 중심 답변

### 질문
{question}

### 답변 지침
정보보안 관리체계 관점에서 체계적으로 한국어로만 답변하세요.

답변:"""
        
        return templates
    
    def _build_korean_examples(self) -> Dict[str, Dict]:
        """한국어 예시"""
        examples = {
            "mc_personal": {
                "question": "개인정보보호법상 개인정보의 정의로 가장 적절한 것은?",
                "answer": "2",
                "reasoning": "개인정보보호법 제2조에 따르면 살아 있는 개인에 관한 정보"
            },
            "mc_electronic": {
                "question": "전자금융거래의 정의로 옳은 것은?",
                "answer": "2",
                "reasoning": "전자적 장치를 통한 금융상품 및 서비스 제공"
            },
            "subj_security": {
                "question": "개인정보보호 관리체계 구축 방안을 설명하시오.",
                "answer": "개인정보보호 관리체계는 다음과 같이 구축합니다. 첫째, 개인정보보호 정책을 수립하고 최고경영진의 책임 하에 운영합니다. 둘째, 개인정보보호 책임자를 지정하고 전담 조직을 구성합니다. 셋째, 개인정보 생명주기별 관리 절차를 수립합니다. 넷째, 기술적 관리적 물리적 안전성 확보조치를 구현합니다. 다섯째, 정기적 점검과 지속적 개선을 수행합니다."
            }
        }
        return examples
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Dict, structure: Dict) -> str:
        """프롬프트 생성"""
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{question[:100]}{question_type}".encode()).hexdigest()[:16]
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        if question_type == "multiple_choice":
            prompt = self._create_mc_prompt(question, analysis, structure)
        else:
            prompt = self._create_subj_prompt(question, analysis, structure)
        
        # 캐시 저장
        self.prompt_cache[cache_key] = prompt
        
        # 통계 업데이트
        self._update_stats(analysis)
        
        return prompt
    
    def _create_mc_prompt(self, question: str, analysis: Dict, structure: Dict) -> str:
        """객관식 프롬프트 생성"""
        
        # 부정형 확인
        if structure.get("has_negative", False):
            negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
            keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
            
            prompt = self.templates["mc_negative"].format(
                question=question,
                keyword=keyword
            )
            self.stats["template_usage"]["mc_negative"] = self.stats["template_usage"].get("mc_negative", 0) + 1
        else:
            prompt = self.templates["mc_basic"].format(question=question)
            self.stats["template_usage"]["mc_basic"] = self.stats["template_usage"].get("mc_basic", 0) + 1
        
        return prompt
    
    def _create_subj_prompt(self, question: str, analysis: Dict, structure: Dict) -> str:
        """주관식 프롬프트 생성"""
        
        # 도메인 특화
        domains = analysis.get("domain", ["일반"])
        
        if "개인정보보호" in domains:
            prompt = self.templates["subj_personal_info"].format(question=question)
            self.stats["template_usage"]["subj_personal_info"] = self.stats["template_usage"].get("subj_personal_info", 0) + 1
        elif "전자금융" in domains:
            prompt = self.templates["subj_electronic"].format(question=question)
            self.stats["template_usage"]["subj_electronic"] = self.stats["template_usage"].get("subj_electronic", 0) + 1
        elif "정보보안" in domains:
            prompt = self.templates["subj_security"].format(question=question)
            self.stats["template_usage"]["subj_security"] = self.stats["template_usage"].get("subj_security", 0) + 1
        else:
            prompt = self.templates["subj_basic"].format(question=question)
            self.stats["template_usage"]["subj_basic"] = self.stats["template_usage"].get("subj_basic", 0) + 1
        
        return prompt
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        """한국어 강화 프롬프트 생성"""
        
        if question_type == "multiple_choice":
            prompt = f"""### 시스템 메시지
당신은 한국의 금융보안 전문가입니다.
절대로 한자, 영어, 일본어 등 외국어를 사용하지 마세요.

### 문제
{question}

### 답변 규칙
1. 숫자 1, 2, 3, 4, 5 중 하나만 선택
2. 추가 설명 없이 숫자만 출력
3. 한국어 이외의 언어 사용 금지

정답:"""
        else:
            prompt = f"""### 시스템 메시지
당신은 한국의 금융보안 전문가입니다.
모든 답변을 순수 한국어로만 작성하세요.
한자(漢字), 영어(English) 등 외국어는 절대 사용하지 마세요.

### 질문
{question}

### 답변 규칙
1. 80자 이상 600자 이내
2. 관련 법령 언급
3. 구체적 방안 제시
4. 순수 한국어만 사용

답변:"""
        
        return prompt
    
    def create_few_shot_prompt(self, question: str, question_type: str,
                             analysis: Dict, num_examples: int = 1) -> str:
        """Few-shot 프롬프트 생성"""
        
        prompt_parts = ["다음은 한국어 금융보안 문제 예시입니다.\n"]
        
        if question_type == "multiple_choice":
            if "개인정보" in question:
                example = self.examples["mc_personal"]
            elif "전자금융" in question:
                example = self.examples["mc_electronic"]
            else:
                example = self.examples["mc_personal"]
            
            prompt_parts.append(f"예시 문제: {example['question']}")
            prompt_parts.append(f"정답: {example['answer']}\n")
        else:
            example = self.examples["subj_security"]
            prompt_parts.append(f"예시 질문: {example['question']}")
            prompt_parts.append(f"답변: {example['answer']}\n")
        
        prompt_parts.append(f"현재 문제:\n{question}")
        prompt_parts.append("위 예시처럼 한국어로만 답하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 최적화"""
        
        # 한국어 강제 접두사 추가
        korean_prefix = "### 중요: 반드시 한국어로만 답변하세요 ###\n\n"
        
        if "solar" in model_name.lower():
            # SOLAR 모델용
            optimized = f"{korean_prefix}### User:\n{prompt}\n\n### Assistant:\n"
        elif "llama" in model_name.lower():
            # Llama 모델용
            optimized = f"{korean_prefix}<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            optimized = korean_prefix + prompt
        
        return optimized
    
    def _update_stats(self, analysis: Dict):
        """통계 업데이트"""
        # 도메인 분포
        domains = analysis.get("domain", ["일반"])
        for domain in domains:
            self.stats["domain_distribution"][domain] = self.stats["domain_distribution"].get(domain, 0) + 1
    
    def get_stats_report(self) -> Dict:
        """통계 보고서"""
        total_prompts = sum(self.stats["template_usage"].values())
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "template_usage": self.stats["template_usage"],
            "domain_distribution": self.stats["domain_distribution"]
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self.stats["template_usage"]:
            print(f"\n=== 프롬프트 통계 ===")
            print(f"총 생성: {sum(self.stats['template_usage'].values())}개")
            print(f"캐시 히트: {self.stats['cache_hits']}회")
            
            if self.stats["template_usage"]:
                most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
                print(f"주요 템플릿: {most_used[0]} ({most_used[1]}회)")
        
        self.prompt_cache.clear()
        self.template_cache.clear()