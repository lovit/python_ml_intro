def print_name(name, transform=None):
    if (transform is not None) and callable(transform):
        name = transform(name)
    print(f'Hello {name}')
