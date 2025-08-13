from nlusystem import ImprovedNLUSystem



def main():
    system = ImprovedNLUSystem(db_path="company.db", model="mistral:latest")
    system.interactive_mode()

if __name__ == "__main__":
    main()
