poems_file = 'poems_original.txt'
poems = []

with open(poems_file, "r", encoding='utf-8',) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or 'F' in content or '2' in content or '￣' in content or 'p' in content or 'ē' in content\
                    or 'ń' in content or '(' in content or '（' in content or '《' in content or '[' in content\
                    or 'Ｃ' in content or '】' in content or 'ｗ' in content:
                continue
            if len(content) < 24 or len(content) > 79:
                continue
            content = '[' + content + ']\n'
            poems.append(content)
        except Exception as e:
            pass


cleaned_poems = open('./poems.txt', mode='w', encoding='utf-8')
cleaned_poems.writelines(poems)
