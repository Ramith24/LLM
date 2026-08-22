#!/usr/bin/env python3
"""
Normalize Zawgyi-encoded Burmese text to Unicode using myanmar-tools library.
This is the production-ready version.
"""
import os
import shutil

def try_import_myanmar_tools():
    try:
        from myanmar import language
        return language
    except ImportError:
        try:
            import myanmar.tools as tools
            return tools
        except ImportError:
            return None

def normalize_with_tools(text, tools_module):
    """Use myanmar-tools to convert Zawgyi to Unicode"""
    # Detect encoding first
    if hasattr(tools_module, 'detect_encoding'):
        encoding = tools_module.detect_encoding(text)
        if encoding == 'zawgyi':
            if hasattr(tools_module, 'zawgyi_to_unicode'):
                return tools_module.zawgyi_to_unicode(text)
            elif hasattr(tools_module, 'convert'):
                return tools_module.convert(text, 'zawgyi', 'unicode')
    elif hasattr(tools_module, 'zawgyi_to_unicode'):
        # Just try conversion
        return tools_module.zawgyi_to_unicode(text)
    return text

def preprocess_file(input_path, output_path, tools_module):
    print(f"Processing {input_path} -> {output_path}")
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue
                # Remove BPE markers first
                line = line.replace("@@ ", "")
                # Convert Zawgyi to Unicode
                normalized = normalize_with_tools(line, tools_module)
                f_out.write(normalized + '\n')
                if i % 10000 == 0 and i > 0:
                    print(f"  Processed {i} lines...")

def main():
    tools = try_import_myanmar_tools()
    if tools is None:
        print("myanmar-tools not installed. Install with: pip install myanmar-tools")
        print("Falling back to heuristic normalization...")
        # Fallback to heuristic
        import subprocess
        subprocess.run(["python", "mt/normalize_zawgyi.py"])
        return
    
    print(f"Using myanmar-tools: {tools}")
    
    # Backup original files first
    for split in ['train', 'val', 'test']:
        src = f"Data/{split}.my.bpe"
        if os.path.exists(src):
            backup = f"Data/{split}.my.bpe.zawgyi_backup"
            if not os.path.exists(backup):
                shutil.copy2(src, backup)
                print(f"Backed up {src} to {backup}")
    
    # Process files
    for split in ['train', 'val', 'test']:
        src = f"Data/{split}.my.bpe"
        if os.path.exists(src):
            temp = f"Data/{split}.my.bpe.unicode"
            preprocess_file(src, temp, tools)
            os.replace(temp, src)
            print(f"Normalized {src}")

if __name__ == "__main__":
    main()