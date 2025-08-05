# prompt_engineering.py
"""
프롬프트 엔지니어링 시스템
"""

import re
import random
import hashlib
import json
from typing import Dict, List, Tuple, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    """프롬프트 엔지니어링 클래스"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.expert_examples = self._build_examples()
        self.adaptive_templates = self._build_templates()
        self.domain_contexts = self._build_domain_contexts()
        
        # 성능 캐시
        self.prompt_cache = {}
        self.template_cache = {}
        
        # 프롬프트 최적화 통계
        self.optimization_stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {},
            "performance_metrics": []
        }
        
    def _build_examples(self) -> Dict[str, List[Dict]]:
        """전문가 예시"""
        examples = {
            "multiple_choice": [
                {
                    "question": """전자금융거래법상 전자금융거래의 정의로 가장 적절한 것은?
1 인터넷을 이용한 모든 거래
2 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래  
3 신용카드를 이용한 거래
4 ATM을 통한 현금 인출
5 모바일 앱을 통한 모든 서비스""",
                    "analysis": """전자금융거래법 제2조 제1호에 따르면 "전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래"가 정의입니다.
선택지 검토:
1번 - 인터넷만 한정(X)
2번 - 법적 정의와 일치(O)
3번 - 신용카드만 한정(X)
4번 - 현금인출만 한정(X)
5번 - 금융과 무관 포함(X)

정답: 2""",
                    "answer": "2",
                    "domain": "전자금융",
                    "complexity": "medium"
                },
                {
                    "question": """개인정보보호법상 개인정보 유출 신고에 관하여 틀린 것은?
1 개인정보처리자는 개인정보가 유출된 사실을 안 때에는 지체 없이 개인정보보호위원회에 신고하여야 한다
2 유출 통지는 개인정보 유출 사실을 안 날부터 72시간 이내에 하여야 한다
3 유출된 개인정보의 항목을 포함하여 신고하여야 한다
4 개인정보 유출로 인한 피해 확산 방지를 위한 대책을 포함하여야 한다
5 고의 또는 중대한 과실로 신고하지 않으면 과태료가 부과될 수 있다""",
                    "analysis": """개인정보보호법 제34조에 따른 유출 신고 의무를 묻는 부정형 문제입니다.
각 선택지 검토:
1번 - 지체 없이 신고 의무 (O)
2번 - 72시간이 아닌 지체 없이 (X)
3번 - 유출 항목 포함 신고 (O)
4번 - 피해 확산 방지 대책 포함 (O)
5번 - 과태료 부과 가능 (O)

정답: 2""",
                    "answer": "2",
                    "domain": "개인정보보호",
                    "complexity": "high"
                }
            ],
            "subjective": [
                {
                    "question": "금융기관의 개인정보보호 관리체계 구축 시 고려해야 할 핵심 요소들을 설명하시오.",
                    "answer": "개인정보보호 관리체계 구축 시 다음 요소들이 핵심입니다. 첫째, 개인정보보호 정책 수립과 시행 체계입니다. 둘째, 개인정보 처리 현황 파악과 정기 점검입니다. 셋째, 접근권한 관리 및 접근통제 시스템입니다. 넷째, 개인정보의 암호화입니다. 다섯째, 접근기록 보관 및 점검입니다.",
                    "domain": "개인정보보호",
                    "complexity": "medium"
                },
                {
                    "question": "전자금융거래에서 접근매체의 안전한 관리를 위한 방안을 설명하시오.",
                    "answer": "접근매체의 안전한 관리를 위해서는 다음과 같은 방안이 필요합니다. 첫째, 접근매체의 선정 시 충분한 보안성을 갖춘 것을 선택해야 합니다. 둘째, 접근매체의 발급, 배송, 전달 과정에서 위조나 변조를 방지해야 합니다. 셋째, 이용자에게 접근매체의 안전한 사용법과 관리 방법을 교육해야 합니다. 넷째, 접근매체의 분실이나 도난 시 즉시 이용정지 조치를 취할 수 있는 체계를 구축해야 합니다.",
                    "domain": "전자금융",
                    "complexity": "high"
                }
            ]
        }
        return examples
    
    def _build_templates(self) -> Dict[str, str]:
        """적응형 템플릿 구축"""
        templates = {}
        
        # 객관식 템플릿들
        templates["mc_simple"] = """문제: {question}

위 문제의 정답 번호를 선택하세요.

정답:"""

        templates["mc_analytical"] = """당신은 금융보안 전문가입니다.

{context}

문제: {question}

단계별 분석:
1. 핵심 개념 파악
2. 각 선택지 검토
3. 근거 기반 판단

정답:"""

        templates["mc_negative"] = """부정형 문제

문제: {question}

이 문제는 '{negative_keyword}'를 찾는 문제입니다.
각 선택지를 신중히 검토하여 명확히 틀린 것을 찾으세요.

정답:"""

        templates["mc_domain_specific"] = """{domain} 전문가 관점에서 다음 문제를 분석하세요.

{domain_context}

문제: {question}

{domain} 관련 법령과 원칙을 바탕으로 정답을 선택하세요.

정답:"""

        # 주관식 템플릿들
        templates["subj_concise"] = """질문: {question}

위 질문에 대해 핵심 내용을 중심으로 명확하고 구체적으로 답변하세요.

답변:"""

        templates["subj_comprehensive"] = """{domain} 전문가로서 다음 질문에 답변하세요.

{context}

질문: {question}

다음 구조로 체계적으로 답변하세요:
1. 주요 개념 설명
2. 구체적 방안/절차
3. 관련 법령/규정
4. 실무적 고려사항

답변:"""

        templates["subj_case_based"] = """실무 상황을 고려하여 답변하세요.

질문: {question}

{case_context}

위 상황을 고려하여 실용적이고 구체적인 답변을 제시하세요.

답변:"""

        return templates
    
    def _build_domain_contexts(self) -> Dict[str, Dict]:
        """도메인별 컨텍스트"""
        contexts = {
            "개인정보보호": {
                "overview": "개인정보보호법과 관련 규정에 따른 개인정보의 안전한 처리와 관리",
                "key_principles": ["최소수집", "목적명확", "동의필수", "안전관리"],
                "main_laws": ["개인정보보호법", "신용정보법"],
                "focus_areas": ["수집·이용", "제공", "파기", "안전성 확보조치"]
            },
            "전자금융": {
                "overview": "전자금융거래법에 따른 안전하고 신뢰할 수 있는 전자금융거래 환경 조성",
                "key_principles": ["전자적 장치", "안전성", "이용자 보호", "책임 분담"],
                "main_laws": ["전자금융거래법", "전자서명법"],
                "focus_areas": ["접근매체", "거래내역", "손실부담", "보안대책"]
            },
            "정보보안": {
                "overview": "정보자산의 기밀성, 무결성, 가용성 확보를 위한 체계적 보안 관리",
                "key_principles": ["위험관리", "접근통제", "지속적 개선", "사고대응"],
                "main_laws": ["정보통신망법", "개인정보보호법"],
                "focus_areas": ["정보보호관리체계", "취약점 관리", "침해사고 대응"]
            },
            "암호화": {
                "overview": "암호화 기술을 통한 정보의 기밀성과 무결성 보장",
                "key_principles": ["기밀성", "무결성", "인증", "부인방지"],
                "main_laws": ["전자서명법", "정보통신망법"],
                "focus_areas": ["대칭키", "공개키", "해시함수", "전자서명"]
            }
        }
        return contexts
    
    def create_adaptive_prompt(self, question: str, question_type: str, 
                             analysis: Dict, strategy: str = "auto") -> str:
        """적응형 프롬프트 생성"""
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{question}{question_type}{strategy}".encode()).hexdigest()[:16]
        if cache_key in self.prompt_cache:
            self.optimization_stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        # 전략 자동 결정
        if strategy == "auto":
            strategy = self._determine_optimal_strategy(question, analysis)
        
        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_adaptive_mc_prompt(question, analysis, strategy)
        else:
            prompt = self._create_adaptive_subjective_prompt(question, analysis, strategy)
        
        # 캐시 저장
        self.prompt_cache[cache_key] = prompt
        
        # 통계 업데이트
        self._update_optimization_stats(strategy, analysis)
        
        return prompt
    
    def _determine_optimal_strategy(self, question: str, analysis: Dict) -> str:
        """최적 전략 결정"""
        complexity = analysis.get("complexity", 0.5)
        domains = analysis.get("domain", [])
        has_negative = analysis.get("has_negative", False)
        
        # 부정형 문제
        if has_negative:
            return "negative_focused"
        
        # 복잡도 기반
        if complexity > 0.7:
            return "comprehensive"
        elif complexity < 0.3:
            return "simple"
        
        # 도메인 특화
        if len(domains) == 1 and domains[0] in self.domain_contexts:
            return "domain_specific"
        
        # 기본 분석형
        return "analytical"
    
    def _create_adaptive_mc_prompt(self, question: str, analysis: Dict, strategy: str) -> str:
        """적응형 객관식 프롬프트"""
        
        if strategy == "negative_focused":
            # 부정형 특화
            negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
            detected_keyword = next((kw for kw in negative_keywords if kw in question), "해당하지 않는")
            
            return self.adaptive_templates["mc_negative"].format(
                question=question,
                negative_keyword=detected_keyword
            )
        
        elif strategy == "domain_specific":
            # 도메인 특화
            domains = analysis.get("domain", [])
            if domains and domains[0] in self.domain_contexts:
                domain = domains[0]
                domain_info = self.domain_contexts[domain]
                
                context = f"핵심 원칙: {', '.join(domain_info['key_principles'])}\n"
                context += f"관련 법령: {', '.join(domain_info['main_laws'])}"
                
                return self.adaptive_templates["mc_domain_specific"].format(
                    domain=domain,
                    domain_context=context,
                    question=question
                )
        
        elif strategy == "simple":
            # 간단한 문제
            return self.adaptive_templates["mc_simple"].format(question=question)
        
        else:
            # 분석형 (기본)
            context = self._generate_contextual_knowledge(question, analysis)
            return self.adaptive_templates["mc_analytical"].format(
                context=context,
                question=question
            )
    
    def _create_adaptive_subjective_prompt(self, question: str, analysis: Dict, strategy: str) -> str:
        """적응형 주관식 프롬프트"""
        
        domains = analysis.get("domain", [])
        complexity = analysis.get("complexity", 0.5)
        
        if strategy == "comprehensive" and complexity > 0.6:
            # 포괄적 답변 요구
            domain = domains[0] if domains and domains[0] in self.domain_contexts else "일반"
            context = ""
            
            if domain in self.domain_contexts:
                domain_info = self.domain_contexts[domain]
                context = f"{domain_info['overview']}\n"
                context += f"주요 영역: {', '.join(domain_info['focus_areas'])}"
            
            return self.adaptive_templates["subj_comprehensive"].format(
                domain=domain,
                context=context,
                question=question
            )
        
        elif strategy == "domain_specific" and domains:
            # 도메인 특화 답변
            domain = domains[0]
            if domain in self.domain_contexts:
                case_context = self._generate_case_context(domain, question)
                return self.adaptive_templates["subj_case_based"].format(
                    question=question,
                    case_context=case_context
                )
        
        # 간결형 (기본)
        return self.adaptive_templates["subj_concise"].format(question=question)
    
    def _generate_contextual_knowledge(self, question: str, analysis: Dict) -> str:
        """맥락적 지식 생성"""
        domains = analysis.get("domain", [])
        
        if not domains:
            return "금융보안 관련 법령과 원칙을 바탕으로 분석하세요."
        
        context_parts = []
        for domain in domains[:2]:  # 최대 2개 도메인
            if domain in self.domain_contexts:
                domain_info = self.domain_contexts[domain]
                context_parts.append(f"{domain}: {domain_info['overview']}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _generate_case_context(self, domain: str, question: str) -> str:
        """사례 컨텍스트 생성"""
        case_contexts = {
            "개인정보보호": "금융기관에서 고객의 개인정보를 처리하는 실무 상황을 고려하여",
            "전자금융": "전자금융거래 서비스를 제공하는 금융기관의 입장에서",
            "정보보안": "금융기관의 정보보안 담당자 관점에서",
            "암호화": "안전한 금융거래를 위한 암호화 시스템 구축 관점에서"
        }
        
        return case_contexts.get(domain, "실무 환경에서")
    
    def create_few_shot_prompt(self, question: str, question_type: str, 
                             analysis: Dict, num_examples: int = 1) -> str:
        """Few-shot 프롬프트 생성"""
        
        # 관련 예시 선택
        examples = self._select_relevant_examples(question, analysis, question_type, num_examples)
        
        if not examples:
            # 예시가 없으면 기본 프롬프트
            return self.create_adaptive_prompt(question, question_type, analysis)
        
        # Few-shot 구성
        prompt_parts = ["다음은 금융보안 문제 해결 예시입니다.\n"]
        
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"예시 {i}:")
            prompt_parts.append(f"문제: {example['question']}")
            if 'analysis' in example:
                prompt_parts.append(f"분석: {example['analysis']}")
            prompt_parts.append(f"답: {example['answer']}\n")
        
        prompt_parts.append("현재 문제:")
        prompt_parts.append(f"문제: {question}")
        prompt_parts.append("위 예시를 참고하여 체계적으로 분석하고 답하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def _select_relevant_examples(self, question: str, analysis: Dict, 
                                question_type: str, num_examples: int) -> List[Dict]:
        """관련 예시 선택"""
        
        available_examples = self.expert_examples.get(question_type, [])
        if not available_examples:
            return []
        
        # 도메인 매칭
        question_domains = set(analysis.get("domain", []))
        
        scored_examples = []
        for example in available_examples:
            score = 0
            
            # 도메인 매칭 점수
            example_domain = example.get("domain", "")
            if example_domain in question_domains:
                score += 3
            
            # 복잡도 매칭 점수
            example_complexity = example.get("complexity", "medium")
            question_complexity = analysis.get("complexity", 0.5)
            
            complexity_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
            example_score = complexity_map.get(example_complexity, 0.5)
            
            if abs(example_score - question_complexity) < 0.3:
                score += 2
            
            # 키워드 매칭
            question_words = set(question.lower().split())
            example_words = set(example['question'].lower().split())
            common_words = question_words & example_words
            score += min(len(common_words) / 10, 1)
            
            scored_examples.append((score, example))
        
        # 점수순 정렬 후 상위 선택
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:num_examples]]
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 프롬프트 최적화"""
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{prompt[:50]}{model_name}".encode()).hexdigest()[:12]
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
        optimized_prompt = prompt
        
        if "solar" in model_name.lower():
            # SOLAR 모델 최적화
            optimized_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
        
        elif "llama" in model_name.lower():
            # Llama 모델 최적화
            optimized_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        elif "gemma" in model_name.lower():
            # Gemma 모델 최적화
            optimized_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # 캐시 저장
        self.template_cache[cache_key] = optimized_prompt
        
        return optimized_prompt
    
    def create_chain_of_thought_prompt(self, question: str, question_type: str, analysis: Dict) -> str:
        """Chain-of-Thought 프롬프트 생성"""
        
        domains = analysis.get("domain", [])
        complexity = analysis.get("complexity", 0.5)
        
        if question_type == "multiple_choice":
            thought_steps = [
                "1. 문제 핵심 개념 파악",
                "2. 관련 법령 및 원칙 확인",
                "3. 각 선택지 세부 분석",
                "4. 논리적 근거 기반 정답 결정"
            ]
            
            if analysis.get("has_negative", False):
                thought_steps[3] = "4. 명확히 틀린 선택지 식별"
            
            prompt = f"""문제: {question}

다음 단계로 체계적으로 사고하세요:
{chr(10).join(thought_steps)}

단계별 분석:"""
            
            return prompt
        
        else:
            # 주관식용 사고 체계
            if complexity > 0.6:
                thought_structure = [
                    "1. 핵심 개념과 정의 명확화",
                    "2. 관련 법령과 규정 검토",
                    "3. 구체적 방안과 절차 도출",
                    "4. 실무적 고려사항 포함",
                    "5. 종합적 결론 도출"
                ]
            else:
                thought_structure = [
                    "1. 핵심 내용 파악",
                    "2. 관련 규정 확인",
                    "3. 구체적 방안 제시"
                ]
            
            domain_focus = ""
            if domains and domains[0] in self.domain_contexts:
                domain_info = self.domain_contexts[domains[0]]
                domain_focus = f"\n{domains[0]} 관점에서 {domain_info['overview']}"
            
            prompt = f"""질문: {question}{domain_focus}

다음 구조로 체계적으로 답변하세요:
{chr(10).join(thought_structure)}

단계별 답변:"""
            
            return prompt
    
    def _update_optimization_stats(self, strategy: str, analysis: Dict):
        """최적화 통계 업데이트"""
        # 템플릿 사용 통계
        if strategy not in self.optimization_stats["template_usage"]:
            self.optimization_stats["template_usage"][strategy] = 0
        self.optimization_stats["template_usage"][strategy] += 1
        
        # 도메인 분포 통계
        domains = analysis.get("domain", ["일반"])
        for domain in domains:
            if domain not in self.optimization_stats["domain_distribution"]:
                self.optimization_stats["domain_distribution"][domain] = 0
            self.optimization_stats["domain_distribution"][domain] += 1
    
    def get_optimization_report(self) -> Dict:
        """최적화 보고서 생성"""
        total_requests = sum(self.optimization_stats["template_usage"].values())
        
        report = {
            "total_prompts_generated": total_requests,
            "cache_hit_rate": self.optimization_stats["cache_hits"] / max(total_requests, 1),
            "most_used_strategy": max(self.optimization_stats["template_usage"].items(), 
                                    key=lambda x: x[1])[0] if self.optimization_stats["template_usage"] else "none",
            "domain_coverage": len(self.optimization_stats["domain_distribution"]),
            "template_efficiency": {
                strategy: count / total_requests 
                for strategy, count in self.optimization_stats["template_usage"].items()
            } if total_requests > 0 else {}
        }
        
        return report
    
    def cleanup(self):
        """리소스 정리"""
        # 통계 출력
        if self.optimization_stats["template_usage"]:
            print(f"\n=== 프롬프트 엔지니어링 통계 ===")
            print(f"총 생성: {sum(self.optimization_stats['template_usage'].values())}개")
            print(f"캐시 히트: {self.optimization_stats['cache_hits']}회")
            
            most_used = max(self.optimization_stats["template_usage"].items(), key=lambda x: x[1])
            print(f"주요 전략: {most_used[0]} ({most_used[1]}회)")
        
        # 캐시 정리
        self.prompt_cache.clear()
        self.template_cache.clear()