"""
准备英语到德语翻译数据集（示例/自定义/iwslt）
"""
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path


def download_iwslt_data(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    iwslt_url = "https://wit3.fbk.eu/archive/2016-01//texts/en/de/en-de.tgz"
    iwslt_file = data_dir / "en-de.tgz"
    if not iwslt_file.exists():
        print(f"下载 IWSLT 数据集...")
        urllib.request.urlretrieve(iwslt_url, iwslt_file)
        print("下载完成")
    extract_dir = data_dir / "iwslt"
    if not extract_dir.exists():
        print("解压数据...")
        with tarfile.open(iwslt_file, "r:gz") as tar:
            tar.extractall(data_dir)
        print("解压完成")
    return extract_dir


def load_iwslt_data(data_dir="data/iwslt"):
    data_dir = Path(data_dir)
    train_en = data_dir / "train.tags.en-de.en"
    train_de = data_dir / "train.tags.en-de.de"
    valid_en = data_dir / "IWSLT16.TED.tst2013.en-de.en.xml"
    valid_de = data_dir / "IWSLT16.TED.tst2013.en-de.de.xml"
    test_en = data_dir / "IWSLT16.TED.tst2014.en-de.en.xml"
    test_de = data_dir / "IWSLT16.TED.tst2014.en-de.de.xml"
    def parse_xml_file(file_path):
        s = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("<seg"):
                    start = line.find(">") + 1
                    end = line.rfind("<")
                    if start < end:
                        s.append(line[start:end].strip())
        return s
    def parse_text_file(file_path):
        s = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("<"):
                    s.append(line)
        return s
    train_src = parse_text_file(train_en) if train_en.exists() else []
    train_tgt = parse_text_file(train_de) if train_de.exists() else []
    valid_src = parse_xml_file(valid_en) if valid_en.exists() else []
    valid_tgt = parse_xml_file(valid_de) if valid_de.exists() else []
    test_src = parse_xml_file(test_en) if test_en.exists() else []
    test_tgt = parse_xml_file(test_de) if test_de.exists() else []
    min_train = min(len(train_src), len(train_tgt))
    train_src = train_src[:min_train]
    train_tgt = train_tgt[:min_train]
    min_valid = min(len(valid_src), len(valid_tgt))
    valid_src = valid_src[:min_valid]
    valid_tgt = valid_tgt[:min_valid]
    min_test = min(len(test_src), len(test_tgt))
    test_src = test_src[:min_test]
    test_tgt = test_tgt[:min_test]
    return {"train": (train_src, train_tgt), "valid": (valid_src, valid_tgt), "test": (test_src, test_tgt)}


def create_simple_dataset(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    examples = [
        ("Hello, how are you?", "Hallo, wie geht es dir?"),
        ("I am fine, thank you.", "Mir geht es gut, danke."),
        ("What is your name?", "Wie heißt du?"),
        ("My name is John.", "Mein Name ist John."),
        ("Good morning.", "Guten Morgen."),
        ("Good evening.", "Guten Abend."),
        ("Good night.", "Gute Nacht."),
        ("Thank you very much.", "Vielen Dank."),
        ("You are welcome.", "Bitte schön."),
        ("I love you.", "Ich liebe dich."),
        ("See you tomorrow.", "Bis morgen."),
        ("Have a nice day.", "Schönen Tag noch."),
        ("How much does it cost?", "Wie viel kostet es?"),
        ("Where is the bathroom?", "Wo ist die Toilette?"),
        ("I don't understand.", "Ich verstehe nicht."),
        ("Can you help me?", "Kannst du mir helfen?"),
        ("I am sorry.", "Es tut mir leid."),
        ("Excuse me.", "Entschuldigung."),
        ("What time is it?", "Wie spät ist es?"),
        ("I am hungry.", "Ich habe Hunger."),
    ]
    train_src, train_tgt = [], []
    for _ in range(100):
        for en, de in examples:
            train_src.append(en)
            train_tgt.append(de)
    (data_dir / "train.en").write_text("\n".join(train_src), encoding="utf-8")
    (data_dir / "train.de").write_text("\n".join(train_tgt), encoding="utf-8")
    valid_src = train_src[:100]
    valid_tgt = train_tgt[:100]
    test_src = train_src[100:200]
    test_tgt = train_tgt[100:200]
    (data_dir / "valid.en").write_text("\n".join(valid_src), encoding="utf-8")
    (data_dir / "valid.de").write_text("\n".join(valid_tgt), encoding="utf-8")
    (data_dir / "test.en").write_text("\n".join(test_src), encoding="utf-8")
    (data_dir / "test.de").write_text("\n".join(test_tgt), encoding="utf-8")
    return {"train": (train_src, train_tgt), "valid": (valid_src, valid_tgt), "test": (test_src, test_tgt)}


def load_custom_data(data_dir="data"):
    data_dir = Path(data_dir)
    def load_file(file_path):
        if not file_path.exists():
            return []
        return [line.strip() for line in file_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    train_src = load_file(data_dir / "train.en")
    train_tgt = load_file(data_dir / "train.de")
    valid_src = load_file(data_dir / "valid.en")
    valid_tgt = load_file(data_dir / "valid.de")
    test_src = load_file(data_dir / "test.en")
    test_tgt = load_file(data_dir / "test.de")
    return {"train": (train_src, train_tgt), "valid": (valid_src, valid_tgt), "test": (test_src, test_tgt)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="准备英语-德语翻译数据")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--source", type=str, default="simple", choices=["simple", "iwslt", "custom"])
    args = ap.parse_args()
    if args.source == "simple":
        print("创建示例数据集...")
        data = create_simple_dataset(args.data_dir)
        print(f"训练集: {len(data['train'][0])} 对")
        print(f"验证集: {len(data['valid'][0])} 对")
        print(f"测试集: {len(data['test'][0])} 对")
    elif args.source == "iwslt":
        print("下载 IWSLT 数据...")
        download_iwslt_data(args.data_dir)
        data = load_iwslt_data(args.data_dir)
        print(f"训练集: {len(data['train'][0])} 对")
        print(f"验证集: {len(data['valid'][0])} 对")
        print(f"测试集: {len(data['test'][0])} 对")
    else:
        print("加载自定义数据...")
        data = load_custom_data(args.data_dir)
        print(f"训练集: {len(data['train'][0])} 对")
        print(f"验证集: {len(data['valid'][0])} 对")
        print(f"测试集: {len(data['test'][0])} 对")




