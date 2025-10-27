from pydantic import BaseModel, Field, create_model

def make_class_model(name: str, methods: dict[str, type[BaseModel]], class_flag: bool = True) -> type[BaseModel]:
    """
    Dynamically builds a Pydantic model that groups multiple method models as submodels.
    """
    fields = {}
    for m_name, m_model in methods.items():
        fields[m_name] = (m_model, Field(default_factory=m_model))
    if class_flag:
        fields["is_class_model"] = (bool, Field(default=True, description="Indicates this is a class-level model"))
    return create_model(f"{name}Model", **fields)
