"""General Section Parser for DOCX elements preserving XML sequence order."""
import re
from typing import List, Dict, Any, Optional

ROMAN_REGEX = re.compile(r"^\s*([IVXLCDM]+)(?:\.|\))\s+(.+)$", re.IGNORECASE)
NUMBERED_REGEX = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.|\))?\s+(.+)$")
STEP_REGEX = re.compile(r"^\s*(Bước\s+\d+|B\d+)(?:\.|\:)?\s*(.*)$", re.IGNORECASE)

class SectionNode:
    def __init__(self, number: str, level: int, title: str, parent: Optional['SectionNode'] = None):
        self.number = number
        self.level = level
        self.title = title
        self.parent = parent
        self.elements: List[Any] = []
        self.children: List['SectionNode'] = []

    def full_path(self) -> str:
        if self.parent and self.parent.title:
            p = self.parent.full_path()
            return f"{p} > {self.title}" if p else self.title
        return self.title

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "level": self.level,
            "title": self.title,
            "elements_count": len(self.elements),
            "children": [c.to_dict() for c in self.children]
        }

class SectionParser:
    """Parses flat list of DOCX elements into a structured hierarchy of SectionNodes."""
    
    @staticmethod
    def is_heading(elem: Any) -> tuple[bool, str, int, str]:
        if elem.kind == "heading":
            title = elem.text.strip()
            level = getattr(elem, "level", 1)
            level = level if level > 0 else 1
            m_num = NUMBERED_REGEX.match(title)
            if m_num:
                num, t = m_num.groups()
                lvl = len(num.split('.'))
                return True, num, lvl, t.strip()
            m_rom = ROMAN_REGEX.match(title)
            if m_rom:
                num, t = m_rom.groups()
                return True, num, 1, t.strip()
            return True, "", level, title

        text = elem.text.strip()
        if not text or len(text) > 120 or elem.kind not in ["paragraph", "numbered", "bullet"]:
            return False, "", 0, ""

        m_num = NUMBERED_REGEX.match(text)
        if m_num:
            num, t = m_num.groups()
            lvl = len(num.split('.'))
            return True, num, lvl, t.strip()

        m_rom = ROMAN_REGEX.match(text)
        if m_rom:
            num, t = m_rom.groups()
            return True, num, 1, t.strip()

        m_step = STEP_REGEX.match(text)
        if m_step:
            step_num, t = m_step.groups()
            return True, step_num, 2, text

        return False, "", 0, ""

    def parse(self, elements: List[Any]) -> SectionNode:
        root = SectionNode(number="0", level=0, title="")
        current = root
        stack = [root]

        for elem in elements:
            is_h, num, lvl, title = self.is_heading(elem)
            if is_h and title:
                while len(stack) > 1 and stack[-1].level >= lvl:
                    stack.pop()

                parent = stack[-1]
                node = SectionNode(number=num, level=lvl, title=title, parent=parent)
                parent.children.append(node)
                stack.append(node)
                current = node
            else:
                current.elements.append(elem)

        return root
