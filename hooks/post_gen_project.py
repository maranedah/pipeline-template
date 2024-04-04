import subprocess
import os


def run_terraform_apply():
    try:
        # Change directory to 'terraform'
        os.chdir("terraform")

        # Run 'terraform init' command
        subprocess.run(["terraform", "init"], check=True)

        # Run 'terraform apply -auto-approve' command
        subprocess.run(["terraform", "apply", "-auto-approve"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Terraform commands: {e}")
        print(f"Command output:\n{e.output.decode()}")


if __name__ == "__main__":
    pass
    # run_terraform_apply()
