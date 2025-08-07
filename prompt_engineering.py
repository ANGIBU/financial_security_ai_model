# prompt_engineering.py
"""
프롬프트 엔지니어링 시스템
"""

import re
import hashlib
from typing import Dict, List, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    """프롬프트 엔지니어링 클래스"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_templates()
        self.examples = self._build_examples()
        
        # 캐시
        self.prompt_cache = {}
        self.template_cache = {}
        
        # 통계
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {}
        }
    
    def _build_templates(self) -> Dict[str, str]:
        """템플릿 구축 (한국어 강제)"""
        templates = {}
        
        # 객관식 템플릿 - 한국어 강제 추가
        templates["mc_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 중요 지침
반드시 한국어로만 답변하세요. 한자나 영어는 사용하지 마세요.

### 문제
{question}

각 선택지를 신중히 검토하고 정답을 선택하세요.
한국어로 간단히 분석한 후 정답 번호만 제시하세요.

정답 번호:"""

        templates["mc_negative"] = """당신은 한국의 금융보안 전문가입니다.

### 중요 지침  
반드시 한국어로만 답변하세요. 한자나 영어는 사용하지 마세요.

### 문제
{question}

이 문제는 '{keyword}'를 찾는 문제입니다.
각 선택지를 검토하여 해당하지 않거나 틀린 것을 찾으세요.

정답 번호:"""

        templates["mc_domain"] = """당신은 한국의 {domain} 전문가입니다.

### 중요 지침
반드시 한국어로만 답변하세요. 한자나 영어 단어는 사용하지 마세요.

### 관련 지식
{context}

### 문제
{question}

{domain} 관점에서 한국어로 분석하여 정답을 선택하세요.

정답 번호:"""

        # 주관식 템플릿 - 한국어 강제 추가
        templates["subj_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 중요 지침
반드시 한국어로만 답변하세요. 한자, 영어 단어, 외국어는 일체 사용하지 마세요.
전문적이고 정확한 한국어로 설명해주세요.

### 질문
{question}

### 답변 지침
- 핵심 내용을 명확하게 설명
- 구체적인 방안 제시  
- 관련 법령이나 규정 언급
- 순수 한국어만 사용

답변:"""

        templates["subj_domain"] = """당신은 한국의 {domain} 전문가입니다.

### 중요 지침
반드시 한국어로만 답변하세요. 한자, 영어, 기타 외국어는 절대 사용하지 마세요.

### 배경 지식
{context}

### 질문
{question}

### 답변 구조
1. 개념 설명 (한국어로)
2. 구체적 방안 (한국어로)
3. 법적 근거 (한국어로)
4. 실무 적용 (한국어로)

모든 내용을 순수 한국어로 작성해주세요.

답변:"""
        
        return templates
    
    def _build_examples(self) -> Dict[str, List[Dict]]:
        """예시 구축 (한국어 강화)"""
        examples = {
            "mc_personal_info": {
                "question": "개인정보보호법상 개인정보의 정의로 가장 적절한 것은?\n1. 모든 정보\n2. 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보\n3. 기업정보\n4. 공개된 정보\n5. 암호화된 정보",
                "answer": "2",
                "reasoning": "개인정보보호법 제2조 제1호에 따르면 개인정보는 살아 있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보입니다."
            },
            "mc_electronic": {
                "question": "전자금융거래의 정의로 옳은 것은?\n1. 모든 금융거래\n2. 전자적 장치를 통한 금융상품 및 서비스 제공과 이용 거래\n3. 인터넷뱅킹만\n4. 신용카드 거래만\n5. 현금거래",
                "answer": "2", 
                "reasoning": "전자금융거래법 제2조에 따르면 전자금융거래는 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래입니다."
            },
            "subj_security": {
                "question": "금융기관의 개인정보보호 관리체계 구축 방안을 설명하시오.",
                "answer": "금융기관의 개인정보보호 관리체계는 다음과 같이 구축해야 합니다. 첫째, 개인정보보호 정책 및 지침을 수립하고 최고경영진의 책임 하에 운영합니다. 둘째, 개인정보보호 책임자와 담당자를 지정하여 전담 조직을 구성합니다. 셋째, 개인정보의 수집과 이용, 제공, 파기 등 전 생명주기에 걸친 관리 절차를 수립합니다. 넷째, 암호화, 접근통제 등 기술적 관리적 물리적 안전성 확보조치를 구현합니다. 다섯째, 정기적인 점검과 감사를 통해 지속적으로 개선합니다."
            }
        }
        return examples
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Dict, structure: Dict) -> str:
        """프롬프트 생성 (한국어 강제)"""
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{question[:100]}{question_type}".encode()).hexdigest()[:16]
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        if question_type == "multiple_choice":
            prompt = self._create_mc_prompt(question, analysis, structure)
        else:
            prompt = self._create_subj_prompt(question, analysis, structure)
        
        # 한국어 강제 추가
        prompt = self._add_korean_enforcement(prompt, question_type)
        
        # 캐시 저장
        self.prompt_cache[cache_key] = prompt
        
        # 통계 업데이트
        self._update_stats(analysis)
        
        return prompt
    
    def _add_korean_enforcement(self, prompt: str, question_type: str) -> str:
        """한국어 사용 강제 추가"""
        
        if question_type == "multiple_choice":
            korean_suffix = """

### 주의사항
- 반드시 한국어로만 답변하세요
- 한자나 영어는 사용하지 마세요
- 간단한 분석 후 정답 번호를 제시하세요"""
        else:
            korean_suffix = """

### 주의사항  
- 반드시 한국어로만 답변하세요
- 한자, 영어, 기타 외국어는 절대 사용하지 마세요
- 전문적이고 명확한 한국어로 설명하세요
- 법령명도 한국어로 표기하세요"""
        
        return prompt + korean_suffix
    
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
        
        # 도메인 특화
        elif analysis.get("domain") and analysis["domain"][0] != "일반":
            domain = analysis["domain"][0]
            context = self._generate_domain_context(domain)
            
            prompt = self.templates["mc_domain"].format(
                domain=domain,
                context=context,
                question=question
            )
            self.stats["template_usage"]["mc_domain"] = self.stats["template_usage"].get("mc_domain", 0) + 1
        
        # 기본
        else:
            prompt = self.templates["mc_basic"].format(question=question)
            self.stats["template_usage"]["mc_basic"] = self.stats["template_usage"].get("mc_basic", 0) + 1
        
        return prompt
    
    def _create_subj_prompt(self, question: str, analysis: Dict, structure: Dict) -> str:
        """주관식 프롬프트 생성"""
        
        # 도메인 특화
        if analysis.get("domain") and analysis["domain"][0] != "일반":
            domain = analysis["domain"][0]
            context = self._generate_domain_context(domain)
            
            prompt = self.templates["subj_domain"].format(
                domain=domain,
                context=context,
                question=question
            )
            self.stats["template_usage"]["subj_domain"] = self.stats["template_usage"].get("subj_domain", 0) + 1
        
        # 기본
        else:
            prompt = self.templates["subj_basic"].format(question=question)
            self.stats["template_usage"]["subj_basic"] = self.stats["template_usage"].get("subj_basic", 0) + 1
        
        return prompt
    
    def _generate_domain_context(self, domain: str) -> str:
        """도메인 컨텍스트 생성 (한국어만)"""
        contexts = {
            "개인정보보호": "개인정보보호법에 따라 개인정보는 정보주체의 동의 하에 수집하고 이용해야 하며, 안전성 확보조치를 통해 보호해야 합니다.",
            "전자금융": "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성과 신뢰성을 확보해야 합니다.",
            "정보보안": "정보보호관리체계를 통해 체계적인 보안 관리와 지속적 개선이 필요합니다.",
            "암호화": "중요 정보는 안전한 암호 기법으로 암호화하여 기밀성과 무결성을 보장해야 합니다.",
            "법령": "관련 법령과 규정을 준수하여 금융보안을 강화해야 합니다."
        }
        
        return contexts.get(domain, "금융보안 원칙에 따라 적절한 조치가 필요합니다.")
    
    def create_few_shot_prompt(self, question: str, question_type: str,
                             analysis: Dict, num_examples: int = 1) -> str:
        """Few-shot 프롬프트 생성 (한국어 강제)"""
        
        prompt_parts = ["다음은 한국어로 작성된 금융보안 문제 예시입니다.\n"]
        
        # 관련 예시 선택
        if question_type == "multiple_choice":
            if "개인정보" in question:
                example = self.examples["mc_personal_info"]
            elif "전자금융" in question:
                example = self.examples["mc_electronic"]
            else:
                example = self.examples["mc_personal_info"]
            
            prompt_parts.append(f"예시 문제:\n{example['question']}")
            prompt_parts.append(f"분석: {example['reasoning']}")
            prompt_parts.append(f"정답: {example['answer']}\n")
        
        else:
            example = self.examples["subj_security"]
            prompt_parts.append(f"예시 질문:\n{example['question']}")
            prompt_parts.append(f"답변:\n{example['answer']}\n")
        
        prompt_parts.append(f"현재 문제:\n{question}")
        prompt_parts.append("위 예시를 참고하여 반드시 한국어로만 답하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답 번호:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 최적화 (한국어 강제 유지)"""
        
        if "solar" in model_name.lower():
            # SOLAR 모델용
            optimized = f"### User:\n{prompt}\n\n### Assistant:\n"
        elif "llama" in model_name.lower():
            # Llama 모델용
            optimized = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            optimized = prompt
        
        return optimized
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        """한국어 강화 프롬프트 생성"""
        
        base_instruction = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 반드시 한국어로만 답변하세요
2. 한자, 중국어, 영어는 절대 사용하지 마세요  
3. 전문 용어도 한국어로 표현하세요
4. 법령명도 한국어로 작성하세요

### 예시 (올바른 표현)
- 개인정보 (O) vs. 個人情報 (X) 
- 소프트웨어 (O) vs. software (X)
- 금융거래 (O) vs. 金融交易 (X)
- 보안 (O) vs. security (X)"""

        if question_type == "multiple_choice":
            specific_instruction = """
### 문제
{question}

위 문제를 한국어로 분석하고 정답 번호만 제시하세요.

정답:"""
        else:
            specific_instruction = """
### 질문  
{question}

위 질문에 대해 전문적이고 체계적으로 한국어로만 답변하세요.

답변:"""
        
        full_prompt = base_instruction + specific_instruction.format(question=question)
        return full_prompt
    
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