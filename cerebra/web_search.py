"""
Модуль для веб-поиска и получения информации из интернета
Файл: web_search.py - Веб-поиск через DuckDuckGo для получения актуальной информации
"""
import requests
import logging
from typing import Optional, List
import time

try:
    from cerebra.logger_config import logger
except ImportError:
    logger = logging.getLogger('cerebra')


class WebSearcher:
    """Класс для поиска информации в интернете"""
    
    def __init__(self):
        self.search_enabled = True
        
    def search(self, query: str, max_results: int = 5) -> List[dict]:
        """
        Поиск в интернете через DuckDuckGo (бесплатный, без API ключа)
        
        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            
        Returns:
            Список словарей с информацией о результатах
        """
        if not self.search_enabled:
            return []
        
        try:
            from bs4 import BeautifulSoup
            
            # Используем DuckDuckGo HTML поиск (не требует API ключа)
            url = "https://html.duckduckgo.com/html/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            params = {'q': query}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Парсим результаты с помощью BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Ищем результаты поиска
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:max_results]:
                try:
                    # Заголовок и ссылка
                    title_elem = div.find('a', class_='result__a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title:
                        results.append({
                            'title': title[:200],
                            'url': url,
                            'snippet': snippet[:300]
                        })
                except Exception as e:
                    logger.debug(f"Ошибка при парсинге результата: {e}")
                    continue
            
            if results:
                logger.info(f"Найдено {len(results)} результатов для запроса: {query}")
            else:
                logger.warning(f"Результаты не найдены для запроса: {query}")
            
            return results
            
        except ImportError:
            logger.warning("BeautifulSoup недоступен, используется упрощенный поиск")
            return self._fallback_search(query)
        except requests.RequestException as e:
            logger.warning(f"Ошибка при поиске в интернете: {e}")
            return self._fallback_search(query)
        except Exception as e:
            logger.error(f"Неожиданная ошибка при поиске: {e}", exc_info=True)
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[dict]:
        """Резервный поиск (мок-данные)"""
        logger.info("Используется резервный режим поиска")
        return [
            {
                'title': f'Результаты поиска для "{query}"',
                'url': 'https://example.com',
                'snippet': f'Информация по запросу "{query}" доступна в интернете.'
            }
        ]
    
    def get_answer_from_web(self, question: str) -> Optional[str]:
        """
        Получить ответ на вопрос из интернета
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Ответ на основе поиска или None
        """
        # Определяем, нужен ли поиск в интернете
        search_keywords = ['что такое', 'кто такой', 'когда', 'где', 'как работает', 
                          'информация о', 'новости', 'погода', 'курс', 'расписание']
        
        needs_search = any(keyword in question.lower() for keyword in search_keywords)
        
        if not needs_search:
            return None
        
        results = self.search(question, max_results=3)
        
        if not results:
            return None
        
        # Формируем ответ на основе результатов
        answer = f"На основе поиска в интернете:\n\n"
        for i, result in enumerate(results, 1):
            answer += f"{i}. {result.get('title', 'Без названия')}\n"
            if result.get('snippet'):
                answer += f"   {result['snippet']}\n"
            answer += f"   Источник: {result.get('url', 'Неизвестно')}\n\n"
        
        return answer.strip()


# Глобальный экземпляр
web_searcher = WebSearcher()

