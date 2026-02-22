import difflib

def get_changed_part(ref_text, new_text):
    """
    计算 new_text 相对于 ref_text 的变更内容。
    逻辑：寻找 ref_text 的后缀与 new_text 的前缀的最大重叠，寻找 new_text 从前到后与 ref_text 出现不相同之后的内容。
    """
    
    # 1. 边界情况：如果你传来的新文本是空的，那新增部分自然也是空的
    if not new_text:
        return ""
        
    # 2. 边界情况：如果参考文本是空的，那所有新文本都是新增的
    if not ref_text:
        return new_text

    # 3. 核心逻辑：寻找重叠 (Overlap Detection)
    # 我们只关心 ref_text 的尾部和 new_text 的头部是否重叠
    # 重叠长度不可能超过 ref_text 的长度，也不可能超过 new_text 的长度
    max_overlap = min(len(ref_text), len(new_text))
    
    # 从最大可能的重叠长度开始尝试，递减到 1
    for i in range(max_overlap, 0, -1):
        # 取 ref_text 的最后 i 个字符
        suffix = ref_text[-i:]
        
        # 检查 new_text 是否以这个 suffix 开头
        if new_text.startswith(suffix):
            # 找到了重叠锚点！
            # new_text[i:] 就是重叠部分之后的新增内容
            return new_text[i:]
        
    # 从头部开始匹配直到出现不同
    for i in range(0, max_overlap, 1):
        if new_text[i] != ref_text[i]:
            return new_text[i:]
            
    # 4. 兜底逻辑：没有发现任何重叠
    # 意味着 new_text 与 ref_text 的尾部完全不相关，视为全新内容
    return new_text

if __name__ == '__main__':
    # --- 测试案例 ---

    # 案例 1：标准重叠 (你的核心需求)
    # text1: ZABC (Z是干扰，ABC是重叠)
    # text2: ABCDEF (ABC是重叠，DEF是新增)
    t1 = "ZABC"
    t2 = "ABCDEF"
    print(f"案例1 ({t1} -> {t2}):\n变更文本: '{get_changed_part(t1, t2)}'\n")

    # 案例 2：完全追加
    t3 = "Hello"
    t4 = "Hello World"
    print(f"案例2 ({t3} -> {t4}):\n变更文本: '{get_changed_part(t3, t4)}'\n")

    # 案例 3：中间插入（应该返回空，因为尾部没变）
    t5 = "Start End"
    t6 = "Start Middle End" 
    # 这里 'End' 是 equal，所以倒序检查第一步就 break 了
    print(f"案例3 ({t5} -> {t6}):\n变更文本: '{get_changed_part(t5, t6)}'\n")

    # 案例 4：尾部被修改
    t7 = "Version 1.0"
    t8 = "Version 2.0 Beta"
    print(f"案例4 ({t7} -> {t8}):\n变更文本: '{get_changed_part(t7, t8)}'")

    # 案例 5：空文本
    t9 = "ABC"
    t10 = ""
    print(f"案例5 ({t9} -> {t10}):\n变更文本: '{get_changed_part(t9, t10)}'")

    # 案例 6：空文本
    t11 = ""
    t12 = "ABC"
    print(f"案例6 ({t11} -> {t12}):\n变更文本: '{get_changed_part(t11, t12)}'")
    
    # 案例 6：头部相同
    t13 = "ではではちょっと新衣装で初めてのウトアクということでね今日は何を着ようかなと思います"
    t14 = "ではではちょっと新衣装で初めての歌悪ということでね今日は何を着ようかなと迷った結果選ば"
    print(f"案例7 ({t13} -> {t14}):\n变更文本: '{get_changed_part(t13, t14)}'")